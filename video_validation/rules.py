from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .config import ValidationConfig
from .models import FrameAnalysis, ValidationEvent


@dataclass(frozen=True)
class RuleDefinition:
    rule_id: str
    title: str
    severity: str
    minimum_duration: float
    predicate: Callable[[FrameAnalysis], bool]
    details: str


class ValidationStrategy:
    """Strategy interface: each rule independently evaluates sampled frame data."""

    def build_rules(self, config: ValidationConfig) -> list[RuleDefinition]:
        raise NotImplementedError


class HandCountStrategy(ValidationStrategy):
    def build_rules(self, config: ValidationConfig) -> list[RuleDefinition]:
        rules = [
            RuleDefinition(
                "possible_hand_missing", "Expected hand(s) not detected", "REVIEW",
                config.maximum_hand_missing_seconds,
                lambda frame: len(frame.hands) < config.expected_hands,
                "Fewer hands than expected were detected. This can mean a hand left the frame, was occluded, or was missed by the model.",
            ),
            RuleDefinition(
                "extra_hand_detected", "More hands than expected", "FAIL",
                config.maximum_extra_hand_seconds,
                lambda frame: len(frame.hands) > config.expected_hands,
                "The detector found more hands than the configured expected count. Review the evidence; false detections are possible.",
            ),
        ]
        return rules


class PersonEntryStrategy(ValidationStrategy):
    def build_rules(self, config: ValidationConfig) -> list[RuleDefinition]:
        if not config.enable_person_detection:
            return []
        return [RuleDefinition(
            "additional_person", "Additional person detected", "FAIL", 0.0,
            lambda frame: frame.person_count > config.allowed_people,
            "The person detector found more people than the configured allowed count.",
        )]


class WorkingAreaStrategy(ValidationStrategy):
    def build_rules(self, config: ValidationConfig) -> list[RuleDefinition]:
        if not config.enable_working_area:
            return []
        return [RuleDefinition(
            "hand_outside_working_area", "Hand outside working area", "REVIEW", 0.4,
            lambda frame: frame.hands_outside_working_area > 0,
            "One or more detected hand landmarks were outside the configured work rectangle.",
        )]


class FrameQualityStrategy(ValidationStrategy):
    def build_rules(self, config: ValidationConfig) -> list[RuleDefinition]:
        rules: list[RuleDefinition] = []
        if config.enable_blur_check:
            rules.append(RuleDefinition(
                "blurry_frame", "Video is blurry", "REVIEW", 1.0,
                lambda frame: frame.blur_score is not None and frame.blur_score < config.minimum_blur_score,
                "Laplacian variance is below the configured sharpness threshold.",
            ))
        if config.enable_black_frame_check:
            rules.append(RuleDefinition(
                "black_frame", "Black or near-black frame", "FAIL", 0.5,
                lambda frame: frame.brightness_mean is not None and frame.brightness_mean < config.black_frame_mean_threshold,
                "Average frame brightness is below the configured threshold.",
            ))
        return rules


class RuleFactory:
    """Factory keeps rule registration separate from the validation engine."""

    @staticmethod
    def create(config: ValidationConfig) -> list[RuleDefinition]:
        strategies: list[ValidationStrategy] = [
            HandCountStrategy(), PersonEntryStrategy(),
            WorkingAreaStrategy(), FrameQualityStrategy(),
        ]
        return [rule for strategy in strategies for rule in strategy.build_rules(config)]


def build_events(
    analyses: list[FrameAnalysis],
    config: ValidationConfig,
    rules: list[RuleDefinition],
    sample_interval: float,
) -> list[ValidationEvent]:
    events: list[ValidationEvent] = []
    if not analyses:
        return events

    for rule in rules:
        active: list[FrameAnalysis] = []
        groups: list[list[FrameAnalysis]] = []
        for analysis in analyses:
            if rule.predicate(analysis):
                if active and analysis.timestamp - active[-1].timestamp > sample_interval + config.event_merge_gap_seconds:
                    groups.append(active)
                    active = []
                active.append(analysis)
            elif active:
                if analysis.timestamp - active[-1].timestamp <= sample_interval + config.event_merge_gap_seconds:
                    continue
                groups.append(active)
                active = []
        if active:
            groups.append(active)

        for group in groups:
            start = group[0].timestamp
            end = group[-1].timestamp + sample_interval
            duration = max(0.0, end - start)
            if duration + 1e-9 < rule.minimum_duration:
                continue
            peak = None
            if rule.rule_id == "extra_hand_detected":
                peak = float(max(len(item.hands) for item in group))
            elif rule.rule_id == "additional_person":
                peak = float(max(item.person_count for item in group))
            events.append(ValidationEvent(
                rule.rule_id, rule.title, rule.severity, start, end,
                rule.details, peak,
            ))
    return sorted(events, key=lambda event: (event.start_seconds, event.rule_id))
