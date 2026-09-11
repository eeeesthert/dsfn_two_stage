"""L2 KNN descriptor matching and Lowe filtering."""
from dataclasses import dataclass
from collections import Counter
import cv2
from .sift_features import FeatureSet
@dataclass(frozen=True)
class Match:
    reference_index:int; target_index:int; distance:float; source_type_reference:str; source_type_target:str

def match_features(reference:FeatureSet,target:FeatureSet,config:dict)->tuple[list[Match],list[Match]]:
    if len(reference.features)<2 or len(target.features)<2:return [],[]
    bf=cv2.BFMatcher(cv2.NORM_L2); pairs=bf.knnMatch(target.descriptors,reference.descriptors,k=2)
    raw=[]; good=[]; ratio=float(config['ratio_test'])
    reverse={}
    if config.get('mutual_check',False):
        for p in bf.knnMatch(reference.descriptors,target.descriptors,k=1): reverse[p[0].queryIdx]=p[0].trainIdx
    for pair in pairs:
        if len(pair)<2:continue
        m,n=pair; item=Match(m.trainIdx,m.queryIdx,float(m.distance),reference.features[m.trainIdx].source_type,target.features[m.queryIdx].source_type); raw.append(item)
        if m.distance<ratio*n.distance and (not reverse or reverse.get(m.trainIdx)==m.queryIdx): good.append(item)
    return raw,good

def match_statistics(matches:list[Match])->dict[str,int]:
    return dict(Counter(f'{m.source_type_reference}-{m.source_type_target}' for m in matches))
