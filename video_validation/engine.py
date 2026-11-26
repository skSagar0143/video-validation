from __future__ import annotations
from pathlib import Path
from typing import Callable
import cv2
import numpy as np
from .config import ValidationConfig
from .detectors import MediaPipeHandDetector, YoloSceneDetector
from .models import FrameAnalysis, VideoValidationResult
from .rules import RuleFactory, build_events
from .video_reader import OpenCVVideoReader
ProgressCallback=Callable[[float,str],None]

class VideoValidationEngine:
    """Application service: samples frames, calls detector adapters and evaluates validation strategies."""
    def __init__(self,config:ValidationConfig):
        config.validate()
        self.config=config
        self.reader=OpenCVVideoReader()

    def analyze(self,video_path:str|Path,output_dir:str|Path,progress:ProgressCallback|None=None)->VideoValidationResult:
        video_path,output_dir=Path(video_path),Path(output_dir)
        output_dir.mkdir(parents=True,exist_ok=True)
        metadata=self.reader.inspect(video_path)
        capture=cv2.VideoCapture(str(video_path))
        if not capture.isOpened(): raise ValueError(f"Could not open video: {video_path}")
        hand_detector=None
        scene_detector=None
        writer=None
        annotated_path=output_dir/f"{video_path.stem}_annotated.mp4"
        analyses=[]
        evidence_paths=[]
        fps=metadata.fps if metadata.fps>0 else 25.0
        sample_every=max(1,int(round(fps/self.config.sample_fps)))
        effective_fps=fps/sample_every
        rules=RuleFactory.create(self.config)
        try:
            hand_detector=MediaPipeHandDetector(self.config,max_hands=max(4,self.config.expected_hands+2))
            if self.config.enable_person_detection or self.config.enable_object_detection:
                object_names=self.config.unwanted_object_names if self.config.enable_object_detection else []
                scene_detector=YoloSceneDetector(self.config.person_confidence,self.config.object_confidence,object_names)
            frame_index=0
            sampled=0
            previous_centres=None
            while True:
                ok,frame=capture.read()
                if not ok: break
                if frame_index%sample_every:
                    frame_index+=1
                    continue
                timestamp=frame_index/fps
                hands=hand_detector.detect(frame)
                people,objects,object_boxes=scene_detector.detect(frame) if scene_detector else (0,[],[])
                gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                blur=float(cv2.Laplacian(gray,cv2.CV_64F).var())
                brightness=float(np.mean(gray))
                outside=self._count_outside(hands)
                centres=sorted([(sum(x for x,y in h.landmarks)/len(h.landmarks),
                                 sum(y for x,y in h.landmarks)/len(h.landmarks)) for h in hands])
                motion=None
                if len(centres)==self.config.expected_hands and previous_centres is not None and len(previous_centres)==len(centres):
                    motion=float(np.mean([np.linalg.norm(np.subtract(a,b)) for a,b in zip(centres,previous_centres)]))
                previous_centres=centres if len(centres)==self.config.expected_hands else None
                analysis=FrameAnalysis(timestamp,frame_index,hands,people,blur,brightness,outside,objects,motion)
                analyses.append(analysis)
                annotated=self._annotate(frame.copy(),analysis,object_boxes)
                if self.config.write_annotated_video:
                    if writer is None:
                        height,width=frame.shape[:2]
                        writer=cv2.VideoWriter(str(annotated_path),cv2.VideoWriter_fourcc(*"mp4v"),effective_fps,(width,height))
                        if not writer.isOpened(): raise RuntimeError("Could not initialize annotated MP4 writer; check codec support and disk space.")
                    writer.write(annotated)
                if sampled<20 and (sampled==0 or sampled%max(1,int(self.config.sample_fps*5))==0):
                    evidence=output_dir/f"evidence_{sampled:04d}.jpg"
                    if cv2.imwrite(str(evidence),annotated): evidence_paths.append(str(evidence))
                sampled+=1
                if progress and metadata.frame_count:
                    progress(min(.99,frame_index/metadata.frame_count),f"Analyzing {timestamp:.1f}s")
                frame_index+=1
        finally:
            capture.release()
            if hand_detector is not None: hand_detector.close()
            if writer is not None: writer.release()
        events=build_events(analyses,self.config,rules,1/max(effective_fps,.1))
        fail=sum(e.severity=="FAIL" for e in events)
        review=sum(e.severity=="REVIEW" for e in events)
        if progress: progress(1.0,"Preparing report")
        configured_objects=self.config.unwanted_object_names if self.config.enable_object_detection else []
        return VideoValidationResult(metadata,"FAIL" if fail else "REVIEW" if review else "PASS",
            len(analyses),events,str(annotated_path) if writer is not None and annotated_path.exists() else None,
            evidence_paths,{"fail_events":fail,"review_events":review,"sample_fps":round(effective_fps,2),
            "expected_hands":self.config.expected_hands,"allowed_people":self.config.allowed_people,
            "object_detection_enabled":self.config.enable_object_detection,
            "unwanted_object_names":configured_objects or []})

    def _count_outside(self,hands):
        if not self.config.enable_working_area:return 0
        area=self.config.working_area or {"x":0,"y":0,"width":1,"height":1}
        x1,y1=area["x"],area["y"]; x2,y2=x1+area["width"],y1+area["height"]
        return sum(any(not(x1<=x<=x2 and y1<=y<=y2) for x,y in hand.landmarks) for hand in hands)

    def _annotate(self,frame,analysis,object_boxes):
        height,width=frame.shape[:2]
        for hand in analysis.hands:
            x1,y1,x2,y2=hand.bbox
            cv2.rectangle(frame,(x1,y1),(x2,y2),(40,220,40),2)
            for x,y in hand.landmarks: cv2.circle(frame,(int(x*width),int(y*height)),2,(0,200,255),-1)
            cv2.putText(frame,f"{hand.handedness} {hand.score:.2f}",(x1,max(20,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.5,(40,220,40),2)
        for name,(x1,y1,x2,y2) in object_boxes:
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
            cv2.putText(frame,f"UNWANTED: {name}",(x1,max(20,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,255),2)
        cv2.putText(frame,f"Hands:{len(analysis.hands)} People:{analysis.person_count} Objects:{','.join(analysis.unwanted_objects)} t={analysis.timestamp:.1f}s",
                    (12,28),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,0),2)
        if self.config.enable_working_area:
            a=self.config.working_area or {"x":0,"y":0,"width":1,"height":1}
            cv2.rectangle(frame,(int(a["x"]*width),int(a["y"]*height)),
                (int((a["x"]+a["width"])*width),int((a["y"]+a["height"])*height)),(255,100,0),2)
        return frame
