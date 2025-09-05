# bev_base.py

from abc import ABC, abstractmethod

class BaseBEV(ABC):
    """
    1) load raw data
    2) preprocess into ego frame
    3) rasterize into BEV
    """
    def __init__(self, config: dict):
        self.cfg = config

    def run(self, token: str):
        pass

    @abstractmethod
    def load(self, token: str):
        """e.g. image+calib or pointcloud+calib"""
        pass

    @abstractmethod
    def transform(self, raw):
        """e.g. IPM matrix or point→ego coords"""
        pass

    @abstractmethod
    def make_bev(self, proc):
        """e.g. warpPerspective or occupancy histogram"""
        pass
