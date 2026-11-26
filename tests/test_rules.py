from video_validation.config import ValidationConfig
from video_validation.models import FrameAnalysis, HandDetection
from video_validation.rules import RuleFactory, build_events


def hand():
    return HandDetection("Left", 0.9, [(0.5, 0.5)] * 21, (10, 10, 20, 20))


def test_sustained_missing_hand_creates_review_event():
    config = ValidationConfig(
        enable_person_detection=False,
        enable_blur_check=False,
        enable_black_frame_check=False,
        maximum_hand_missing_seconds=1.0,
    )
    frames = [
        FrameAnalysis(timestamp=t, frame_index=i, hands=[hand()])
        for i, t in enumerate([0.0, 0.5, 1.0, 1.5])
    ]
    events = build_events(frames, config, RuleFactory.create(config), 0.5)
    assert any(event.rule_id == "possible_hand_missing" for event in events)


def test_expected_hand_count_does_not_flag_two_hands():
    config = ValidationConfig(
        enable_person_detection=False,
        enable_blur_check=False,
        enable_black_frame_check=False,
    )
    frames = [
        FrameAnalysis(timestamp=0.0, frame_index=0, hands=[hand(), hand()]),
        FrameAnalysis(timestamp=0.5, frame_index=1, hands=[hand(), hand()]),
    ]
    events = build_events(frames, config, RuleFactory.create(config), 0.5)
    assert events == []
