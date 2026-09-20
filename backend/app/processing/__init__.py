"""公共资料处理流程。"""

from .processor import ProcessingResult, process_extracted_file, process_markdown_file
from .metadata import DocumentMetadata, MetadataError, update_document_metadata
from .knowledge_scope import (
    KnowledgeScopeError,
    document_ids_for_scope,
    filter_vector_rows,
    validate_status_values,
)
from .deletion import DeletionError, DeletionPlan, DeletionResult, DeletionService
from .queue import ProcessingQueue, QueueError, QueueTaskResult
from .web import (
    AttachmentCandidate,
    WebExtractionError,
    WebPageExtraction,
    extract_web_page,
    fetch_web_page,
    select_attachments,
    selected_attachments,
)
from .versions import VersionError, VersionResult, process_with_version_choice

__all__ = [
    "ProcessingQueue",
    "ProcessingResult",
    "DocumentMetadata",
    "MetadataError",
    "KnowledgeScopeError",
    "QueueError",
    "QueueTaskResult",
    "AttachmentCandidate",
    "WebExtractionError",
    "WebPageExtraction",
    "extract_web_page",
    "fetch_web_page",
    "select_attachments",
    "selected_attachments",
    "VersionError",
    "VersionResult",
    "process_with_version_choice",
    "update_document_metadata",
    "document_ids_for_scope",
    "filter_vector_rows",
    "validate_status_values",
    "DeletionError",
    "DeletionPlan",
    "DeletionResult",
    "DeletionService",
    "process_markdown_file",
    "process_extracted_file",
]
