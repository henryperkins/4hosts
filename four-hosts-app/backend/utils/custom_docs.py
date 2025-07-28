"""
Custom API Documentation Utilities
Provides custom OpenAPI schema and documentation UI
"""

from typing import Dict, Any
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi


def custom_openapi(app: FastAPI) -> Dict[str, Any]:
    """Generate custom OpenAPI schema"""
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="Four Hosts Research API",
        version="3.0.0",
        description="""
        # Four Hosts Research API
        
        A paradigm-aware research system that provides AI-powered analysis through four distinct perspectives:
        
        ## Paradigms
        
        - **DOLORES** - Revolutionary perspective focusing on exposing injustices
        - **TEDDY** - Protective perspective emphasizing care and support
        - **BERNARD** - Analytical perspective based on data and evidence
        - **MAEVE** - Strategic perspective for competitive advantage
        
        ## Features
        
        - 🔍 Real-time web research with paradigm-specific search strategies
        - 🤖 AI-powered synthesis and answer generation
        - 📊 Source credibility analysis
        - 🔐 JWT-based authentication with role-based access
        - 📡 WebSocket support for real-time progress updates
        - 🪝 Webhook notifications for async events
        - 📁 Multiple export formats (PDF, JSON, CSV, etc.)
        - 🚦 Rate limiting based on user roles
        - 📈 Prometheus metrics and monitoring
        
        ## Authentication
        
        Use JWT Bearer tokens for authentication. Include the token in the Authorization header:
        ```
        Authorization: Bearer <your-token>
        ```
        
        ## Rate Limits
        
        Rate limits vary by user role:
        - **Free**: 10 req/min, 100 req/hour
        - **Basic**: 30 req/min, 500 req/hour
        - **Pro**: 60 req/min, 1000 req/hour
        - **Enterprise**: 200 req/min, 10000 req/hour
        """,
        routes=app.routes,
        tags=[
            {
                "name": "authentication",
                "description": "User registration, login, and token management",
            },
            {
                "name": "paradigms",
                "description": "Paradigm classification and analysis",
            },
            {"name": "research", "description": "Submit and manage research queries"},
            {"name": "sources", "description": "Source credibility and management"},
            {"name": "webhooks", "description": "Webhook subscription management"},
            {"name": "websockets", "description": "Real-time updates via WebSocket"},
            {
                "name": "export",
                "description": "Export research results in various formats",
            },
            {"name": "system", "description": "System health and statistics"},
            {"name": "monitoring", "description": "Metrics and monitoring endpoints"},
        ],
    )

    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "bearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "JWT authentication token",
        },
        "apiKey": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "API key for service-to-service communication",
        },
    }

    # Add global security
    openapi_schema["security"] = [{"bearerAuth": []}, {"apiKey": []}]

    # Add example requests/responses
    openapi_schema["components"]["examples"] = {
        "research_query": {
            "value": {
                "query": "What are the latest developments in renewable energy technology?",
                "options": {
                    "depth": "standard",
                    "paradigm_override": None,
                    "include_secondary": True,
                    "max_sources": 50,
                    "language": "en",
                    "region": "us",
                    "enable_real_search": True,
                },
            }
        },
        "paradigm_classification": {
            "value": {
                "primary": "bernard",
                "secondary": "maeve",
                "distribution": {
                    "dolores": 0.1,
                    "teddy": 0.15,
                    "bernard": 0.5,
                    "maeve": 0.25,
                },
                "confidence": 0.85,
                "explanation": {
                    "bernard": "Analytical and evidence-based approach",
                    "maeve": "Strategic and action-oriented perspective",
                },
            }
        },
    }

    app.openapi_schema = openapi_schema
    return app.openapi_schema


def get_custom_swagger_ui_html(*, openapi_url: str) -> str:
    """Generate custom Swagger UI HTML"""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Four Hosts Research API - Documentation</title>
        <link rel="stylesheet" type="text/css" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui.css">
        <style>
            body {{
                margin: 0;
                padding: 0;
            }}
            .swagger-ui .topbar {{
                display: none;
            }}
            .swagger-ui .info {{
                margin-bottom: 20px;
            }}
            .swagger-ui .info .title {{
                color: #2d3748;
            }}
            .swagger-ui .scheme-container {{
                background: #f7fafc;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
            }}
        </style>
    </head>
    <body>
        <div id="swagger-ui"></div>
        <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui-bundle.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui-standalone-preset.js"></script>
        <script>
            window.onload = function() {{
                window.ui = SwaggerUIBundle({{
                    url: "{openapi_url}",
                    dom_id: '#swagger-ui',
                    deepLinking: true,
                    presets: [
                        SwaggerUIBundle.presets.apis,
                        SwaggerUIStandalonePreset
                    ],
                    plugins: [
                        SwaggerUIBundle.plugins.DownloadUrl
                    ],
                    layout: "StandaloneLayout",
                    defaultModelsExpandDepth: 1,
                    defaultModelExpandDepth: 1,
                    docExpansion: "list",
                    filter: true,
                    showExtensions: true,
                    showCommonExtensions: true,
                    tryItOutEnabled: true,
                    supportedSubmitMethods: ['get', 'post', 'put', 'delete', 'patch'],
                    onComplete: function() {{
                        console.log("Swagger UI loaded successfully");
                    }}
                }});
            }}
        </script>
    </body>
    </html>
    """


def get_custom_redoc_html(*, openapi_url: str) -> str:
    """Generate custom ReDoc HTML"""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Four Hosts Research API - ReDoc</title>
        <meta charset="utf-8"/>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {{
                margin: 0;
                padding: 0;
            }}
            .menu-content {{
                background: #fafafa;
            }}
            .api-info h1 {{
                color: #2d3748;
            }}
        </style>
    </head>
    <body>
        <div id="redoc-container"></div>
        <script src="https://cdn.jsdelivr.net/npm/redoc@2.1.3/bundles/redoc.standalone.js"></script>
        <script>
            Redoc.init(
                "{openapi_url}",
                {{
                    scrollYOffset: 50,
                    hideHostname: false,
                    hideDownloadButton: false,
                    disableSearch: false,
                    onlyRequiredInSamples: false,
                    theme: {{
                        colors: {{
                            primary: {{
                                main: '#2d3748'
                            }},
                            success: {{
                                main: '#48bb78'
                            }},
                            warning: {{
                                main: '#ed8936'
                            }},
                            error: {{
                                main: '#e53e3e'
                            }}
                        }},
                        typography: {{
                            fontSize: '14px',
                            lineHeight: '1.5',
                            fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
                            headings: {{
                                fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
                                fontWeight: '600'
                            }},
                            code: {{
                                fontSize: '13px',
                                fontFamily: '"SF Mono", Monaco, Consolas, "Courier New", monospace'
                            }}
                        }},
                        sidebar: {{
                            backgroundColor: '#f7fafc',
                            textColor: '#2d3748'
                        }}
                    }},
                    expandResponses: "200,201",
                    requiredPropsFirst: true,
                    sortPropsAlphabetically: false,
                    showExtensions: true,
                    hideSingleRequestSampleTab: false,
                    jsonSampleExpandLevel: 2,
                    hideSchemaTitles: false,
                    simpleOneOfTypeLabel: false,
                    menuToggle: true,
                    nativeScrollbars: false,
                    pathInMiddlePanel: false,
                    untrustedSpec: false
                }},
                document.getElementById('redoc-container')
            );
        </script>
    </body>
    </html>
    """
