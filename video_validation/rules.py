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
    def build_rules(self, config: ValidationConfig) -> list[RuleDefinition]: raise NotImplementedError

class HandCountStrategy(ValidationStrategy):
    def build_rules(self, c):
        return [
            RuleDefinition("possible_hand_missing","Expected hand(s) not detected","REVIEW",c.maximum_hand_missing_seconds,
                lambda f: len(f.hands)<c.expected_hands,
                "Fewer hands than expected were detected. This may be hand exit, occlusion, or a model miss."),
            RuleDefinition("extra_hand_detected","More hands than expected","FAIL",c.maximum_extra_hand_seconds,
                lambda f: len(f.hands)>c.expected_hands,
                "More hands than configured were detected. Inspect evidence for false positives.")
        ]

class PersonEntryStrategy(ValidationStrategy):
    def build_rules(self,c):
        if not c.enable_person_detection: return []
        return [RuleDefinition("additional_person","Additional person detected","FAIL",0.0,
            lambda f:f.person_count>c.allowed_people,"Person count exceeded the configured allowed count.")]

class UnwantedObjectStrategy(ValidationStrategy):
    def build_rules(self,c):
        if not c.enable_object_detection: return []
        return [RuleDefinition("unwanted_object","Unwanted object detected","FAIL",0.0,
            lambda f:bool(f.unwanted_objects),"Detected configured unwanted object class(es): inspect the evidence frame.")]

class HandMotionStrategy(ValidationStrategy):
    def build_rules(self,c):
        if not c.enable_hand_motion_check: return []
        return [RuleDefinition("hands_stationary","Hands appear stationary","REVIEW",c.stationary_hand_seconds,
            lambda f:len(f.hands)>=c.expected_hands and f.hand_motion_score is not None and f.hand_motion_score<c.minimum_hand_motion_score,
            "Hand landmark centroids moved less than the configured threshold. Pauses may be intentional; inspect context.")]

class WorkingAreaStrategy(ValidationStrategy):
    def build_rules(self,c):
        if not c.enable_working_area:return []
        return [RuleDefinition("hand_outside_working_area","Hand outside working area","REVIEW",0.4,
            lambda f:f.hands_outside_working_area>0,"One or more detected hand landmarks were outside the configured work rectangle.")]

class FrameQualityStrategy(ValidationStrategy):
    def build_rules(self,c):
        rules=[]
        if c.enable_blur_check: rules.append(RuleDefinition("blurry_frame","Video is blurry","REVIEW",1.0,
            lambda f:f.blur_score is not None and f.blur_score<c.minimum_blur_score,"Laplacian variance is below the sharpness threshold."))
        if c.enable_black_frame_check: rules.append(RuleDefinition("black_frame","Black or near-black frame","FAIL",0.5,
            lambda f:f.brightness_mean is not None and f.brightness_mean<c.black_frame_mean_threshold,"Average brightness is below the configured threshold."))
        return rules

class RuleFactory:
    @staticmethod
    def create(c):
        strategies=[HandCountStrategy(),PersonEntryStrategy(),UnwantedObjectStrategy(),HandMotionStrategy(),WorkingAreaStrategy(),FrameQualityStrategy()]
        return [r for strategy in strategies for r in strategy.build_rules(c)]

def build_events(analyses:list[FrameAnalysis],config:ValidationConfig,rules:list[RuleDefinition],sample_interval:float)->list[ValidationEvent]:
    events=[]
    for rule in rules:
        groups=[]; active=[]; last_true=None
        for item in analyses:
            if rule.predicate(item):
                if active and last_true is not None and item.timestamp-last_true>sample_interval+config.event_merge_gap_seconds:
                    groups.append(active); active=[]
                active.append(item); last_true=item.timestamp
            elif active and last_true is not None and item.timestamp-last_true>sample_interval+config.event_merge_gap_seconds:
                groups.append(active); active=[]; last_true=None
        if active: groups.append(active)
        for group in groups:
            start=group[0].timestamp; end=group[-1].timestamp+sample_interval
            duration=max(0.0,end-start)
            if duration+1e-9<rule.minimum_duration: continue
            peak=None
            if rule.rule_id=="extra_hand_detected": peak=float(max(len(x.hands) for x in group))
            elif rule.rule_id=="additional_person": peak=float(max(x.person_count for x in group))
            elif rule.rule_id=="hands_stationary": peak=float(min(x.hand_motion_score for x in group if x.hand_motion_score is not None))
            elif rule.rule_id=="unwanted_object": peak=float(len(set(n for x in group for n in x.unwanted_objects)))
            events.append(ValidationEvent(rule.rule_id,rule.title,rule.severity,start,end,rule.details,peak,group[0].frame_index))
    return sorted(events,key=lambda e:(e.start_seconds,e.rule_id))
