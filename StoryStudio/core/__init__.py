"""
StoryStudio Core Pipeline & Management Package
"""
from StoryStudio.core.catalogue_manager import CatalogueManager
from StoryStudio.core.continuity_tracker import ContinuityTracker
from StoryStudio.core.project_manager import ProjectManager
from StoryStudio.core.pipeline import StoryStudioPipeline

__all__ = [
    "CatalogueManager",
    "ContinuityTracker",
    "ProjectManager",
    "StoryStudioPipeline"
]
