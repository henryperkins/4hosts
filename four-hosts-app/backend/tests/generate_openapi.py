"""
Generate OpenAPI/Swagger documentation for Four Hosts Research API
Phase 5: Production-Ready Features
"""

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
import json
from typing import Dict, Any

# API metadata
API_TITLE = "Four Hosts Research API"
API_VERSION = "1.0.0"
API_DESCRIPTION = """
# Four Hosts Research API

RESTful API for paradigm-aware research capabilities based on Westworld host consciousness models.

## Overview

The Four Hosts Research API provides intelligent research capabilities through four distinct paradigms:

- **Dolores** (Revolutionary): Challenges systems, exposes issues, seeks transformation
- **Teddy** (Devotion): Focuses on support, empathy, and helping others
- **Bernard** (Analytical): Data-driven, objective, scientific approach
- **Maeve** (Strategic): Opportunity-focused, tactical, results-oriented

## Key Features

- 🔍 **Paradigm Classification**: Automatically classifies queries into appropriate research paradigms
- 🌐 **Multi-Source Research**: Aggregates information from diverse sources
- 🧠 **Intelligent Synthesis**: Creates coherent answers from multiple perspectives
- 📊 **Real-time Progress**: WebSocket support for live research updates
- 🔐 **Secure Access**: JWT authentication with role-based permissions
- 📈 **Rate Limiting**: Tiered access with usage quotas
- 🪝 **Webhooks**: Async notifications for research events
- 📁 **Export Options**: PDF, JSON, CSV, Excel, and Markdown formats

## Authentication

The API uses Bearer token authentication. Include your API key or JWT token in the Authorization header:

```
Authorization: Bearer YOUR_TOKEN
```

## Rate Limits

Rate limits vary by user role:

| Role | Requests/Min | Requests/Hour | Requests/Day | Concurrent |
|------|--------------|---------------|--------------|------------|
| Free | 10 | 100 | 500 | 1 |
| Basic | 30 | 500 | 5,000 | 3 |
| Pro | 60 | 1,000 | 20,000 | 10 |
| Enterprise | 200 | 10,000 | 100,000 | 50 |

## WebSocket Support

Connect to `/ws` with authentication token for real-time research progress updates.

## Webhook Events

Subscribe to events:
- `research.started`
- `research.progress`
- `research.completed`
- `research.failed`
- `synthesis.completed`
- `export.ready`
"""

# Contact information
CONTACT_INFO = {
    "name": "Four Hosts Research Support",
    "url": "https://fourhoststresearch.com/support",
    "email": "support@fourhoststresearch.com"
}

# License information
LICENSE_INFO = {
    "name": "Commercial License",
    "url": "https://fourhoststresearch.com/license"
}

# Tags for organizing endpoints
TAGS_METADATA = [
    {
        "name": "authentication",
        "description": "User authentication and authorization endpoints",
    },
    {
        "name": "research",
        "description": "Core research query and retrieval endpoints",
    },
    {
        "name": "classification",
        "description": "Paradigm classification endpoints",
    },
    {
        "name": "webhooks",
        "description": "Webhook subscription management",
    },
    {
        "name": "export",
        "description": "Export research results in various formats",
    },
    {
        "name": "user",
        "description": "User profile and settings management",
    },
    {
        "name": "admin",
        "description": "Administrative endpoints (requires admin role)",
    },
    {
        "name": "health",
        "description": "Health checks and system status",
    }
]

# Security schemes
SECURITY_SCHEMES = {
    "bearerAuth": {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "JWT",
        "description": "JWT token or API key authentication"
    },
    "apiKey": {
        "type": "apiKey",
        "in": "header",
        "name": "X-API-Key",
        "description": "API key authentication (alternative to Bearer token)"
    }
}

def custom_openapi(app: FastAPI) -> Dict[str, Any]:
    """Generate custom OpenAPI schema"""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=API_TITLE,
        version=API_VERSION,
        description=API_DESCRIPTION,
        routes=app.routes,
        contact=CONTACT_INFO,
        license_info=LICENSE_INFO,
        tags=TAGS_METADATA
    )
    
    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = SECURITY_SCHEMES
    
    # Add global security requirement
    openapi_schema["security"] = [{"bearerAuth": []}, {"apiKey": []}]
    
    # Add servers
    openapi_schema["servers"] = [
        {
            "url": "https://api.fourhoststresearch.com/v1",
            "description": "Production server"
        },
        {
            "url": "https://staging-api.fourhoststresearch.com/v1",
            "description": "Staging server"
        },
        {
            "url": "http://localhost:8000",
            "description": "Development server"
        }
    ]
    
    # Add external documentation
    openapi_schema["externalDocs"] = {
        "description": "Full API Documentation",
        "url": "https://docs.fourhoststresearch.com/api"
    }
    
    # Add custom schemas
    if "components" not in openapi_schema:
        openapi_schema["components"] = {}
    
    if "schemas" not in openapi_schema["components"]:
        openapi_schema["components"]["schemas"] = {}
    
    # Add custom response schemas
    openapi_schema["components"]["schemas"]["ErrorResponse"] = {
        "type": "object",
        "properties": {
            "error": {"type": "string"},
            "detail": {"type": "string"},
            "code": {"type": "string"},
            "timestamp": {"type": "string", "format": "date-time"}
        },
        "required": ["error", "detail"]
    }
    
    openapi_schema["components"]["schemas"]["RateLimitInfo"] = {
        "type": "object",
        "properties": {
            "limit": {"type": "integer"},
            "remaining": {"type": "integer"},
            "reset": {"type": "string", "format": "date-time"},
            "retry_after": {"type": "integer"}
        }
    }
    
    # Add webhook payload schemas
    openapi_schema["components"]["schemas"]["WebhookPayload"] = {
        "type": "object",
        "properties": {
            "event": {"type": "string"},
            "timestamp": {"type": "string", "format": "date-time"},
            "webhook_id": {"type": "string"},
            "delivery_id": {"type": "string"},
            "data": {"type": "object"},
            "metadata": {"type": "object"}
        },
        "required": ["event", "timestamp", "data"]
    }
    
    # Add common responses
    openapi_schema["components"]["responses"] = {
        "UnauthorizedError": {
            "description": "Authentication required",
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                }
            }
        },
        "ForbiddenError": {
            "description": "Insufficient permissions",
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                }
            }
        },
        "NotFoundError": {
            "description": "Resource not found",
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                }
            }
        },
        "RateLimitError": {
            "description": "Rate limit exceeded",
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/RateLimitInfo"}
                }
            },
            "headers": {
                "X-RateLimit-Limit": {
                    "description": "Rate limit ceiling",
                    "schema": {"type": "integer"}
                },
                "X-RateLimit-Remaining": {
                    "description": "Remaining requests",
                    "schema": {"type": "integer"}
                },
                "X-RateLimit-Reset": {
                    "description": "Reset timestamp",
                    "schema": {"type": "integer"}
                },
                "Retry-After": {
                    "description": "Seconds until retry",
                    "schema": {"type": "integer"}
                }
            }
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

def generate_openapi_json(app: FastAPI, output_path: str = "openapi.json"):
    """Generate OpenAPI JSON file"""
    schema = custom_openapi(app)
    with open(output_path, "w") as f:
        json.dump(schema, f, indent=2)
    print(f"OpenAPI schema saved to {output_path}")

# Custom API documentation pages
def get_custom_swagger_ui_html(
    openapi_url: str = "/openapi.json",
    title: str = API_TITLE
) -> str:
    """Generate custom Swagger UI HTML"""
    return get_swagger_ui_html(
        openapi_url=openapi_url,
        title=title,
        oauth2_redirect_url="/docs/oauth2-redirect",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
        swagger_favicon_url="https://fastapi.tiangolo.com/img/favicon.png"
    )

def get_custom_redoc_html(
    openapi_url: str = "/openapi.json",
    title: str = API_TITLE
) -> str:
    """Generate custom ReDoc HTML"""
    return get_redoc_html(
        openapi_url=openapi_url,
        title=title,
        redoc_js_url="https://cdn.jsdelivr.net/npm/redoc@next/bundles/redoc.standalone.js",
        redoc_favicon_url="https://fastapi.tiangolo.com/img/favicon.png"
    )

# API Examples
API_EXAMPLES = {
    "research_query": {
        "basic": {
            "summary": "Basic research query",
            "value": {
                "query": "How can small businesses compete with Amazon?",
                "options": {
                    "depth": "standard"
                }
            }
        },
        "advanced": {
            "summary": "Advanced research query",
            "value": {
                "query": "What are the ethical implications of AI in healthcare?",
                "options": {
                    "depth": "deep",
                    "paradigm_override": "bernard",
                    "include_secondary": True,
                    "max_sources": 200,
                    "language": "en",
                    "region": "us"
                }
            }
        }
    },
    "webhook_config": {
        "basic": {
            "summary": "Basic webhook configuration",
            "value": {
                "url": "https://example.com/webhook",
                "events": ["research.completed"]
            }
        },
        "advanced": {
            "summary": "Advanced webhook configuration",
            "value": {
                "url": "https://example.com/webhook",
                "events": ["research.started", "research.progress", "research.completed"],
                "secret": "webhook_secret_key",
                "headers": {
                    "X-Custom-Header": "value"
                },
                "retry_policy": {
                    "max_attempts": 5,
                    "initial_delay": 2,
                    "max_delay": 120
                }
            }
        }
    }
}

if __name__ == "__main__":
    # This would be imported from your main FastAPI app
    # For demonstration, create a minimal app
    from fastapi import FastAPI
    
    app = FastAPI(
        title=API_TITLE,
        version=API_VERSION,
        description=API_DESCRIPTION,
        contact=CONTACT_INFO,
        license_info=LICENSE_INFO,
        openapi_tags=TAGS_METADATA
    )
    
    # Generate OpenAPI JSON
    generate_openapi_json(app)