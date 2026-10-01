"""Pipeline — orchestrates Layers 1-6 into one analysis (TZ §3, §18.4).

    ingest -> filter/quality/artifacts -> re-reference -> features
    -> detection -> causal reasoning -> critical scan -> mode gate
    -> LLM interpretation (grounded) -> outputs

Layers are called through their contracts, so any layer can be swapped without
touching the others.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..contracts.causal import CausalGraph
from ..contracts.events import DetectionResult
from ..contracts.report import LLMReport, OperatingMode
from ..contracts.signal import ClinicalContext, PatientInfo, UnifiedSignal
from ..critical.safety import scan_critical_findings
from ..layer1_ingest.registry import ingest
from ..layer2_preprocess.artifacts import ArtifactReport, ICAStep, flag_artifacts
from ..layer2_preprocess.bad_channels import detect_bad_channels
from ..layer2_preprocess.filters import apply_filters
from ..layer2_preprocess.montage_engine import MontageEngine, MontagedSignal
from ..layer2_preprocess.reref import rereference
from ..layer2_preprocess.sqi import compute_sqi, overall_quality
from ..layer3_features.feature_set import FeatureSet, compute_features
from ..layer3_features.norms import NormsEngine
from ..layer4_detect.registry import default_detectors, run_detectors
from ..layer5_reasoning.physiology import PhysiologyEngine
from ..layer5_reasoning.rules_engine import build_causal_graph
from ..layer6_interpret.grounding import validate_grounding
from ..layer6_interpret.llm_base import (
    ModeGateResult,
    build_llm_input,
    evaluate_mode_b_gate,
)
from ..layer6_interpret.factory import make_provider
from ..layer6_interpret.protocol import render_protocol
from ..outputs.causal_plot import plot_causal_graph
from ..outputs.json_out import build_result_json, save_json
from ..outputs.pdf_report import build_pdf_report
from ..outputs.plots import plot_dsa_aeeg, plot_montage_with_events
from ..outputs.topomap import plot_band_topomaps
from .config_loader import ConfigBundle, load_configs


@dataclass
class PipelineOutput:
    signal: UnifiedSignal
    features: FeatureSet
    detection: DetectionResult
    graph: CausalGraph
    report: LLMReport
    gate: ModeGateResult
    montaged: MontagedSignal
    montage_name: str
    artifacts: ArtifactReport
    result_json: dict[str, Any] = field(default_factory=dict)

    @property
    def protocol_text(self) -> str:
        return render_protocol(self.report)


class Pipeline:
    def __init__(
        self,
        config: ConfigBundle | None = None,
        provider_pref: str = "auto",
        run_ica: bool = True,
        calibration_file: str | Path | None = None,
    ):
        self.cfg = config or load_configs()
        self.provider_pref = provider_pref
        self.run_ica = run_ica
        self.norms = NormsEngine(self.cfg.norms)
        self.physiology = PhysiologyEngine(self.norms)
        self.calibrator = None
        if calibration_file and Path(calibration_file).exists():
            from ..calibration.calibrator import ConfidenceCalibrator

            self.calibrator = ConfidenceCalibrator.load(calibration_file)

    # -- entry points ----------------------------------------------------
    def analyze_file(
        self,
        path: str | Path,
        *,
        mode: OperatingMode = OperatingMode.A_DECISION_SUPPORT,
        montage_name: str = "double_banana",
        patient: PatientInfo | None = None,
        context: ClinicalContext | None = None,
    ) -> PipelineOutput:
        signal = ingest(path, patient=patient, context=context)
        return self.analyze_signal(signal, mode=mode, montage_name=montage_name)

    def analyze_signal(
        self,
        signal: UnifiedSignal,
        *,
        mode: OperatingMode = OperatingMode.A_DECISION_SUPPORT,
        montage_name: str = "double_banana",
    ) -> PipelineOutput:
        cfg = self.cfg

        # --- Layer 2: preprocess / quality / artifacts ---
        filtered = apply_filters(signal, cfg.filters)
        filtered.quality.per_channel_sqi = compute_sqi(filtered, mains_hz=cfg.filters.notch_hz)
        bad = detect_bad_channels(filtered)
        if bad:
            filtered.quality.flags.append(f"bad_channels:{','.join(bad)}")
        artifacts = flag_artifacts(filtered, mains_hz=cfg.filters.notch_hz)
        if self.run_ica:
            filtered = ICAStep().apply(filtered)

        analysis = rereference(filtered, scheme="average")
        analysis.quality.per_channel_sqi = filtered.quality.per_channel_sqi

        signal_quality = overall_quality(
            analysis.quality.per_channel_sqi, analysis.channel_names
        )

        # --- Layer 3: features ---
        features = compute_features(analysis, cfg.filters)

        # --- Layer 4: detection ---
        detectors = default_detectors()
        if analysis.patient.postmenstrual_age_weeks is not None:
            from ..layer4_detect.neonatal import NeonatalBackgroundDetector

            detectors = detectors + [NeonatalBackgroundDetector(self.norms)]
        detection = run_detectors(analysis, features, cfg.thresholds, artifacts, detectors=detectors)

        # --- confidence calibration (TZ §13): calibrated confidences feed the
        #     causal graph, mode-B gate, and alarms so thresholds are reliable ---
        if self.calibrator is not None:
            self.calibrator.apply_to_detection(detection)

        # --- Layer 5: causal reasoning ---
        graph = build_causal_graph(
            detection, analysis.context, analysis.patient,
            cfg.rule_base, self.physiology, artifacts,
        )

        # --- Critical findings (independent of LLM/mode) ---
        critical = scan_critical_findings(detection)

        # --- Mode gate (B) ---
        gate = evaluate_mode_b_gate(
            mode, features, detection, graph, signal_quality,
            artifacts.artifact_fraction, cfg.thresholds,
        )

        # --- Layer 6: interpretation (grounded) ---
        llm_input = build_llm_input(
            mode, analysis.patient, analysis.context, features, detection, graph, gate
        )
        provider = make_provider(self.provider_pref)
        report = provider.generate(llm_input)
        validate_grounding(report, set(llm_input.allowed_grounding_keys))
        report.critical_findings = critical  # always attached

        # --- montage view for plotting/localization ---
        montaged = MontageEngine(analysis).apply(cfg.montage(montage_name))

        result_json = build_result_json(
            analysis, features, detection, graph, report, gate, montage_name
        )
        result_json["calibration"] = {
            "applied": self.calibrator is not None,
            "default_temperature": self.calibrator.default if self.calibrator else 1.0,
            "per_code_temperatures": self.calibrator.temperatures if self.calibrator else {},
            "metrics": self.calibrator.metrics if self.calibrator else {},
        }

        return PipelineOutput(
            signal=analysis,
            features=features,
            detection=detection,
            graph=graph,
            report=report,
            gate=gate,
            montaged=montaged,
            montage_name=montage_name,
            artifacts=artifacts,
            result_json=result_json,
        )

    # -- output helpers --------------------------------------------------
    def save_outputs(self, output: PipelineOutput, out_dir: str | Path) -> dict[str, Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}
        paths["json"] = save_json(output.result_json, out_dir / "result.json")
        proto = out_dir / "protocol_ru_uz.txt"
        proto.write_text(output.protocol_text, encoding="utf-8")
        paths["protocol"] = proto
        try:
            # center the curve view on the most salient event (ictal > any), else first 30s
            window = None
            salient = output.detection.by_group("ictal") or output.detection.events
            if salient:
                ev = salient[0]
                window = (max(0.0, ev.t_start - 5.0), ev.t_start + 25.0)
            paths["curves"] = plot_montage_with_events(
                output.montaged, output.detection.events, out_dir / "curves.png",
                window_s=window,
            )
            if output.features.dsa is not None and output.features.aeeg is not None:
                paths["dsa_aeeg"] = plot_dsa_aeeg(
                    output.features.dsa, output.features.aeeg, out_dir / "dsa_aeeg.png"
                )
            if self.cfg.electrode_coords:
                paths["topomap"] = plot_band_topomaps(
                    output.features, self.cfg.electrode_coords, out_dir / "topomap.png"
                )
            paths["causal_graph"] = plot_causal_graph(
                output.graph, out_dir / "causal_graph.png"
            )
        except Exception as exc:  # plotting must not break the run
            (out_dir / "plot_error.txt").write_text(str(exc), encoding="utf-8")

        # PDF report (embeds whichever figures were produced)
        try:
            paths["pdf"] = build_pdf_report(
                output.report, output.detection, output.signal,
                output.montage_name, paths, out_dir / "report.pdf",
            )
        except Exception as exc:  # report generation must not break the run
            (out_dir / "pdf_error.txt").write_text(str(exc), encoding="utf-8")
        return paths
