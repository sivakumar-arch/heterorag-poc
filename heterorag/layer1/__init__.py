# heterorag/layer1/__init__.py
"""
Layer 1 — Service Discovery

Public API:
    ServiceDescriptor      — the atomic unit of I₁
    ServiceRegistry        — registry + two-stage RelevanceFilter
    I1_ServiceDiscoveryOutput — formal interface I₁
    build_poc_registry     — factory for the three-service POC mesh
"""

from .models import (
    ColumnSpec,
    ConnectionConfig,
    DocumentFieldSpec,
    DocumentSchemaSpec,
    DocumentTypeSpec,
    EdgeTypeSpec,
    GraphSchemaSpec,
    I1_ServiceDiscoveryOutput,
    NodeTypeSpec,
    PersistenceType,
    SQLSchemaSpec,
    ServiceDescriptor,
    TableSpec,
)
from .registry import RelevanceFilter, ServiceRegistry
from .poc_descriptors import (
    CONTENT_SERVICE_DESCRIPTOR,
    KNOWLEDGE_GRAPH_DESCRIPTOR,
    USER_ACTIVITY_DESCRIPTOR,
    build_poc_registry,
)

__all__ = [
    "ColumnSpec",
    "ConnectionConfig",
    "ContentServiceDescriptor",
    "DocumentFieldSpec",
    "DocumentSchemaSpec",
    "DocumentTypeSpec",
    "EdgeTypeSpec",
    "GraphSchemaSpec",
    "I1_ServiceDiscoveryOutput",
    "NodeTypeSpec",
    "PersistenceType",
    "RelevanceFilter",
    "SQLSchemaSpec",
    "ServiceDescriptor",
    "ServiceRegistry",
    "TableSpec",
    "USER_ACTIVITY_DESCRIPTOR",
    "KNOWLEDGE_GRAPH_DESCRIPTOR",
    "CONTENT_SERVICE_DESCRIPTOR",
    "build_poc_registry",
]
