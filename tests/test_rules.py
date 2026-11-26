from video_validation.config import ValidationConfig
from video_validation.models import FrameAnalysis, HandDetection
from video_validation.rules import RuleFactory, build_events

def hand():
    return HandDetection("Left",0.9,[(0.5,0.5)]*21,(10,10,20,20))

def config(**kwargs):
    values=dict(enable_person_detection=False,enable_object_detection=False,enable_blur_check=False,enable_black_frame_check=False)
    values.update(kwargs)
    return ValidationConfig(**values)

def test_sustained_missing_hand_creates_review_event():
    c=config(maximum_hand_missing_seconds=1.0)
    frames=[FrameAnalysis(t,i,[hand()]) for i,t in enumerate([0.0,0.5,1.0,1.5])]
    assert any(e.rule_id=="possible_hand_missing" for e in build_events(frames,c,RuleFactory.create(c),0.5))

def test_expected_hand_count_does_not_flag_two_hands():
    c=config(); frames=[FrameAnalysis(0,0,[hand(),hand()]),FrameAnalysis(.5,1,[hand(),hand()])]
    assert build_events(frames,c,RuleFactory.create(c),.5)==[]

def test_unwanted_object_is_reported():
    c=config(enable_object_detection=True,unwanted_object_names=["cell phone"])
    frames=[FrameAnalysis(0,12,[hand(),hand()],unwanted_objects=["cell phone"])]
    assert any(e.rule_id=="unwanted_object" for e in build_events(frames,c,RuleFactory.create(c),.2))

def test_additional_person_is_reported():
    c=config(enable_person_detection=True,allowed_people=1)
    frames=[FrameAnalysis(0,0,[hand(),hand()],person_count=2)]
    assert any(e.rule_id=="additional_person" for e in build_events(frames,c,RuleFactory.create(c),.2))

def test_invalid_roi_is_rejected():
    c=config(enable_working_area=True,working_area={"x":.8,"y":0,"width":.4,"height":.5})
    try: c.validate()
    except ValueError: return
    raise AssertionError("Invalid ROI should raise ValueError")

def test_motion_rule_can_be_enabled():
    c=config(enable_hand_motion_check=True,stationary_hand_seconds=.4)
    frames=[FrameAnalysis(t,i,[hand(),hand()],hand_motion_score=0.0) for i,t in enumerate([0,.2,.4])]
    assert any(e.rule_id=="hands_stationary" for e in build_events(frames,c,RuleFactory.create(c),.2))
