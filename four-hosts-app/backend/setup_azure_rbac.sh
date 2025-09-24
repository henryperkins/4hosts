#!/bin/bash

# Azure RBAC Setup Script for Content Safety with Azure OpenAI
# This script automates the RBAC configuration following Azure AI Foundry best practices

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration - Update these values
CONTENT_SAFETY_NAME="content-safety-hperkin4"
RESOURCE_GROUP="${RESOURCE_GROUP:-rg-hperkin4-8776}"
OPENAI_RESOURCE_NAME="${OPENAI_RESOURCE_NAME:-}"
SUBSCRIPTION_ID="${SUBSCRIPTION_ID:-fe000daf-8df4-49e4-99d8-6e789060f760}"

echo -e "${GREEN}Azure RBAC Setup for Content Safety${NC}"
echo "======================================"
echo ""

# Function to check if user is logged in
check_azure_login() {
    echo -n "Checking Azure login status... "
    if az account show &>/dev/null; then
        echo -e "${GREEN}✓${NC}"
        CURRENT_SUB=$(az account show --query id -o tsv)
        echo "Current subscription: $CURRENT_SUB"
    else
        echo -e "${RED}✗${NC}"
        echo "Please login to Azure first: az login"
        exit 1
    fi
}

# Function to check if resource exists
check_resource_exists() {
    local resource_name=$1
    local resource_type=$2

    echo -n "Checking if $resource_type '$resource_name' exists... "
    if az cognitiveservices account show -n "$resource_name" -g "$RESOURCE_GROUP" &>/dev/null; then
        echo -e "${GREEN}✓${NC}"
        return 0
    else
        echo -e "${RED}✗${NC}"
        return 1
    fi
}

# Main setup process
main() {
    echo "Configuration:"
    echo "  Content Safety Resource: $CONTENT_SAFETY_NAME"
    echo "  Resource Group: $RESOURCE_GROUP"
    echo "  OpenAI Resource: ${OPENAI_RESOURCE_NAME:-<not set>}"
    echo ""

    # Check Azure login
    check_azure_login

    # Set subscription if needed
    if [ "$SUBSCRIPTION_ID" != "" ]; then
        echo "Setting subscription to: $SUBSCRIPTION_ID"
        az account set --subscription "$SUBSCRIPTION_ID"
    fi

    # Check Content Safety resource exists
    if ! check_resource_exists "$CONTENT_SAFETY_NAME" "Content Safety"; then
        echo -e "${RED}Error: Content Safety resource not found${NC}"
        exit 1
    fi

    # Step 1: Check/Enable Managed Identity
    echo ""
    echo -e "${YELLOW}Step 1: Checking Managed Identity${NC}"
    echo "------------------------------------"

    MI_STATUS=$(az cognitiveservices account show \
        --name "$CONTENT_SAFETY_NAME" \
        --resource-group "$RESOURCE_GROUP" \
        --query "identity.type" -o tsv 2>/dev/null || echo "None")

    if [ "$MI_STATUS" == "SystemAssigned" ]; then
        echo -e "${GREEN}✓${NC} System-assigned managed identity is already enabled"
    else
        echo "Enabling system-assigned managed identity..."
        az cognitiveservices account identity assign \
            --name "$CONTENT_SAFETY_NAME" \
            --resource-group "$RESOURCE_GROUP" >/dev/null
        echo -e "${GREEN}✓${NC} Managed identity enabled"
    fi

    # Get Principal ID
    CS_PRINCIPAL_ID=$(az cognitiveservices account show \
        --name "$CONTENT_SAFETY_NAME" \
        --resource-group "$RESOURCE_GROUP" \
        --query identity.principalId -o tsv)

    echo "Content Safety MI Principal ID: $CS_PRINCIPAL_ID"

    # Step 2: Configure OpenAI access (if resource name provided)
    if [ -n "$OPENAI_RESOURCE_NAME" ]; then
        echo ""
        echo -e "${YELLOW}Step 2: Configuring Azure OpenAI Access${NC}"
        echo "----------------------------------------"

        # Check if OpenAI resource exists
        if ! check_resource_exists "$OPENAI_RESOURCE_NAME" "Azure OpenAI"; then
            echo -e "${RED}Warning: Azure OpenAI resource not found${NC}"
            echo "Skipping OpenAI configuration"
        else
            # Get OpenAI resource ID
            OPENAI_RESOURCE_ID=$(az cognitiveservices account show \
                --name "$OPENAI_RESOURCE_NAME" \
                --resource-group "$RESOURCE_GROUP" \
                --query id -o tsv)

            echo "OpenAI Resource ID: $OPENAI_RESOURCE_ID"

            # Check existing role assignment
            echo -n "Checking existing role assignments... "
            EXISTING_ROLE=$(az role assignment list \
                --assignee "$CS_PRINCIPAL_ID" \
                --scope "$OPENAI_RESOURCE_ID" \
                --query "[?roleDefinitionName=='Cognitive Services OpenAI User'].roleDefinitionName" \
                -o tsv 2>/dev/null || echo "")

            if [ -n "$EXISTING_ROLE" ]; then
                echo -e "${GREEN}✓${NC}"
                echo "Role 'Cognitive Services OpenAI User' already assigned"
            else
                echo -e "${YELLOW}Not found${NC}"
                echo "Assigning 'Cognitive Services OpenAI User' role..."

                az role assignment create \
                    --assignee-object-id "$CS_PRINCIPAL_ID" \
                    --assignee-principal-type ServicePrincipal \
                    --role "Cognitive Services OpenAI User" \
                    --scope "$OPENAI_RESOURCE_ID" >/dev/null

                echo -e "${GREEN}✓${NC} Role assigned successfully"
            fi

            # List deployments
            echo ""
            echo "Available OpenAI deployments:"
            az cognitiveservices account deployment list \
                --name "$OPENAI_RESOURCE_NAME" \
                --resource-group "$RESOURCE_GROUP" \
                --query "[].{name:name,model:properties.model.name,version:properties.model.version}" \
                -o table 2>/dev/null || echo "Unable to list deployments"
        fi
    else
        echo ""
        echo -e "${YELLOW}Step 2: Skipped (No OpenAI resource specified)${NC}"
        echo "To enable reasoning/correction features, set OPENAI_RESOURCE_NAME"
    fi

    # Step 3: Verification
    echo ""
    echo -e "${YELLOW}Step 3: Verification${NC}"
    echo "--------------------"

    # Show Content Safety configuration
    echo "Content Safety Configuration:"
    az cognitiveservices account show \
        --name "$CONTENT_SAFETY_NAME" \
        --resource-group "$RESOURCE_GROUP" \
        --query "{name:name,endpoint:properties.endpoint,identity:identity.type,principalId:identity.principalId}" \
        -o table

    # Show role assignments
    echo ""
    echo "Role Assignments for Content Safety MI:"
    az role assignment list \
        --assignee "$CS_PRINCIPAL_ID" \
        --query "[].{role:roleDefinitionName,scope:scope}" \
        -o table 2>/dev/null || echo "No roles found"

    # Generate .env snippet
    echo ""
    echo -e "${GREEN}Setup Complete!${NC}"
    echo ""
    echo "Add these to your .env file:"
    echo "-----------------------------"

    CS_ENDPOINT=$(az cognitiveservices account show \
        --name "$CONTENT_SAFETY_NAME" \
        --resource-group "$RESOURCE_GROUP" \
        --query properties.endpoint -o tsv)

    cat <<EOF
# Content Safety Configuration
CONTENT_SAFETY_ENDPOINT=$CS_ENDPOINT
CONTENT_SAFETY_API_KEY=<get-from-azure-portal>
EOF

    if [ -n "$OPENAI_RESOURCE_NAME" ]; then
        OPENAI_ENDPOINT=$(az cognitiveservices account show \
            --name "$OPENAI_RESOURCE_NAME" \
            --resource-group "$RESOURCE_GROUP" \
            --query properties.endpoint -o tsv 2>/dev/null || echo "")

        if [ -n "$OPENAI_ENDPOINT" ]; then
            cat <<EOF

# Azure OpenAI Configuration (for reasoning/correction)
AZURE_OPENAI_ENDPOINT=$OPENAI_ENDPOINT
AZURE_OPENAI_API_KEY=<get-from-azure-portal>
AZURE_OPENAI_DEPLOYMENT=<your-deployment-name>
EOF
        fi
    fi

    echo ""
    echo -e "${YELLOW}Note:${NC} Role propagation can take up to 15 minutes."
    echo "      Test with: python diagnose_azure_auth.py"
}

# Run main function
main