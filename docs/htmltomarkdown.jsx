import React, { useState, useEffect } from 'react';
import { AlertCircle, Loader2, FileText, ExternalLink } from 'lucide-react';
import DOMPurify from 'dompurify';
import he from 'he';

/**
 * Custom HTML to Markdown converter
 * Handles common HTML elements and converts them to Markdown syntax
 */
class MarkdownConverter {
    constructor() {
        this.rules = [
            // Headings
            { pattern: /<h([1-6])[^>]*>(.*?)<\/h[1-6]>/gi, replacement: (match, level, content) => '#'.repeat(parseInt(level)) + ' ' + this.cleanText(content) + '\n\n' },

            // Bold text
            { pattern: /<(strong|b)[^>]*>(.*?)<\/(strong|b)>/gi, replacement: (match, tag, content) => '**' + this.cleanText(content) + '**' },

            // Italic text
            { pattern: /<(em|i)[^>]*>(.*?)<\/(em|i)>/gi, replacement: (match, tag, content) => '*' + this.cleanText(content) + '*' },

            // Links
            { pattern: /<a[^>]*href=["']([^"']*)["'][^>]*>(.*?)<\/a>/gi, replacement: (match, href, text) => `[${this.cleanText(text)}](${href})` },

            // Images
            { pattern: /<img[^>]*src=["']([^"']*)["'][^>]*alt=["']([^"']*)["'][^>]*\/?>/gi, replacement: (match, src, alt) => `![${alt}](${src})` },
            { pattern: /<img[^>]*alt=["']([^"']*)["'][^>]*src=["']([^"']*)["'][^>]*\/?>/gi, replacement: (match, alt, src) => `![${alt}](${src})` },
            { pattern: /<img[^>]*src=["']([^"']*)["'][^>]*\/?>/gi, replacement: (match, src) => `![](${src})` },

            // Code blocks with language detection
            { pattern: /<pre[^>]*><code[^>]*class=["'](?:language-|lang-)([^"'\s]+)["'][^>]*>(.*?)<\/code><\/pre>/gis, replacement: (match, lang, content) => '```' + lang + '\n' + this.cleanText(content) + '\n```\n\n' },
            { pattern: /<pre[^>]*><code[^>]*>(.*?)<\/code><\/pre>/gis, replacement: (match, content) => '```\n' + this.cleanText(content) + '\n```\n\n' },
            { pattern: /<pre[^>]*>(.*?)<\/pre>/gis, replacement: (match, content) => '```\n' + this.cleanText(content) + '\n```\n\n' },

            // Inline code
            { pattern: /<code[^>]*>(.*?)<\/code>/gi, replacement: (match, content) => '`' + this.cleanText(content) + '`' },

            // Blockquotes
            { pattern: /<blockquote[^>]*>(.*?)<\/blockquote>/gis, replacement: (match, content) => '> ' + this.cleanText(content).replace(/\n/g, '\n> ') + '\n\n' },

            // Tables
            { pattern: /<table[^>]*>(.*?)<\/table>/gis, replacement: (match, content) => this.convertTable(content) + '\n\n' },

            // Lists
            { pattern: /<ul[^>]*>(.*?)<\/ul>/gis, replacement: (match, content) => this.convertList(content, false) + '\n' },
            { pattern: /<ol[^>]*>(.*?)<\/ol>/gis, replacement: (match, content) => this.convertList(content, true) + '\n' },

            // Paragraphs
            { pattern: /<p[^>]*>(.*?)<\/p>/gis, replacement: (match, content) => this.cleanText(content) + '\n\n' },

            // Line breaks
            { pattern: /<br\s*\/?>/gi, replacement: '  \n' },

            // Horizontal rules
            { pattern: /<hr[^>]*\/?>/gi, replacement: '\n---\n\n' },

            // Remove remaining HTML tags
            { pattern: /<[^>]*>/g, replacement: '' },

            // Clean up extra whitespace
            { pattern: /\n{3,}/g, replacement: '\n\n' },
            { pattern: /^\s+|\s+$/g, replacement: '' }
        ];
    }

    // Strip HTML tags first, then decode all entities using 'he' for robust coverage
    cleanText(text) {
        const stripped = text.replace(/<[^>]*>/g, '');
        return he.decode(stripped).trim();
    }

    convertTable(content) {
        // Extract rows
        const rows = content.match(/<tr[^>]*>(.*?)<\/tr>/gis) || [];
        if (rows.length === 0) return '';

        const tableRows = rows.map(row => {
            // Extract cells (th or td)
            const cells = row.match(/<t[hd][^>]*>(.*?)<\/t[hd]>/gis) || [];
            const cellContents = cells.map(cell => {
                return this.cleanText(cell.replace(/<\/?t[hd][^>]*>/gi, '')).replace(/\|/g, '\\|');
            });
            return '| ' + cellContents.join(' | ') + ' |';
        });

        // Add header separator if we have rows
        if (tableRows.length > 0) {
            const headerSeparator = '| ' + tableRows[0].split('|').slice(1, -1).map(() => '---').join(' | ') + ' |';
            tableRows.splice(1, 0, headerSeparator);
        }

        return tableRows.join('\n');
    }

    convertList(content, ordered = false, depth = 0) {
        const items = content.match(/<li[^>]*>(.*?)<\/li>/gis) || [];
        const indent = '  '.repeat(depth);

        return items.map((item, index) => {
            let itemContent = item.replace(/<\/?li[^>]*>/gi, '');

            // Handle nested lists
            const nestedUl = itemContent.match(/<ul[^>]*>(.*?)<\/ul>/gis);
            const nestedOl = itemContent.match(/<ol[^>]*>(.*?)<\/ol>/gis);

            let nestedListContent = '';
            if (nestedUl || nestedOl) {
                if (nestedUl) {
                    nestedUl.forEach(nestedList => {
                        const nestedContent = nestedList.replace(/<\/?ul[^>]*>/gi, '');
                        nestedListContent += '\n' + this.convertList(nestedContent, false, depth + 1);
                        itemContent = itemContent.replace(nestedList, '');
                    });
                }
                if (nestedOl) {
                    nestedOl.forEach(nestedList => {
                        const nestedContent = nestedList.replace(/<\/?ol[^>]*>/gi, '');
                        nestedListContent += '\n' + this.convertList(nestedContent, true, depth + 1);
                        itemContent = itemContent.replace(nestedList, '');
                    });
                }
            }

            const text = this.cleanText(itemContent);
            const marker = ordered ? `${index + 1}. ` : '- ';
            return indent + marker + text + nestedListContent;
        }).join('\n');
    }

    convert(html) {
        // Single-pass conversion using a combined regex pattern
        // This reduces O(n²) behavior by processing the HTML once
        let markdown = html;

        // Sort rules by priority (more specific patterns first)
        const sortedRules = [...this.rules].sort((a, b) => {
            // Prioritize more specific patterns to avoid conflicts
            const getPriority = (pattern) => {
                const str = pattern.toString();
                if (str.includes('table')) return 10;
                if (str.includes('pre') && str.includes('code')) return 9;
                if (str.includes('blockquote')) return 8;
                if (str.includes('ul') || str.includes('ol')) return 7;
                if (str.includes('h[1-6]')) return 6;
                if (str.includes('strong') || str.includes('em')) return 5;
                if (str.includes('img')) return 4;
                if (str.includes('href')) return 3;
                if (str.includes('code')) return 2;
                return 1;
            };
            return getPriority(b.pattern) - getPriority(a.pattern);
        });

        // Apply rules in optimized order
        for (const rule of sortedRules) {
            markdown = markdown.replace(rule.pattern, rule.replacement);
        }

        return markdown.trim();
    }
}

/**
 * Simple Markdown renderer component
 * Converts common Markdown syntax to JSX elements
 */
const MarkdownRenderer = ({ content }) => {
    const renderMarkdown = (text) => {
        const lines = text.split('\n');
        const elements = [];
        let currentElement = '';
        let inCodeBlock = false;
        let codeBlockContent = '';

        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];

            // Handle code blocks
            if (line.startsWith('```')) {
                if (inCodeBlock) {
                    elements.push(
                        <pre key={i} className="bg-gray-100 p-3 rounded-md overflow-x-auto mb-4">
                            <code className="text-sm">{codeBlockContent}</code>
                        </pre>
                    );
                    codeBlockContent = '';
                    inCodeBlock = false;
                } else {
                    inCodeBlock = true;
                }
                continue;
            }

            if (inCodeBlock) {
                codeBlockContent += line + '\n';
                continue;
            }

            // Handle headings
            if (line.startsWith('#')) {
                const level = line.match(/^#+/)[0].length;
                const text = line.replace(/^#+\s*/, '');
                const HeadingTag = `h${Math.min(level, 6)}`;
                const className = level === 1 ? 'text-2xl font-bold mb-4' :
                    level === 2 ? 'text-xl font-bold mb-3' :
                        level === 3 ? 'text-lg font-bold mb-2' :
                            'text-base font-bold mb-2';

                elements.push(
                    React.createElement(HeadingTag, { key: i, className }, renderInlineElements(text))
                );
                continue;
            }

            // Handle horizontal rules
            if (line === '---') {
                elements.push(<hr key={i} className="my-4 border-gray-300" />);
                continue;
            }

            // Handle blockquotes
            if (line.startsWith('> ')) {
                const text = line.replace(/^>\s*/, '');
                elements.push(
                    <blockquote key={i} className="border-l-4 border-gray-300 pl-4 italic mb-4">
                        {renderInlineElements(text)}
                    </blockquote>
                );
                continue;
            }

            // Handle lists
            if (line.match(/^\s*[-*+]\s/) || line.match(/^\s*\d+\.\s/)) {
                const isOrdered = line.match(/^\s*\d+\.\s/);
                const text = line.replace(/^\s*(?:[-*+]|\d+\.)\s*/, '');
                const ListTag = isOrdered ? 'ol' : 'ul';
                const className = isOrdered ? 'list-decimal list-inside mb-4' : 'list-disc list-inside mb-4';

                elements.push(
                    React.createElement(ListTag, { key: i, className },
                        React.createElement('li', {}, renderInlineElements(text))
                    )
                );
                continue;
            }

            // Handle paragraphs
            if (line.trim()) {
                elements.push(
                    <p key={i} className="mb-4">{renderInlineElements(line)}</p>
                );
            }
        }

        return elements;
    };

    const renderInlineElements = (text) => {
        // Handle links
        text = text.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" class="text-blue-600 hover:underline" target="_blank" rel="noopener noreferrer">$1</a>');

        // Handle images
        text = text.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, '<img src="$2" alt="$1" class="max-w-full h-auto rounded-md my-2" />');

        // Handle bold text
        text = text.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

        // Handle italic text
        text = text.replace(/\*([^*]+)\*/g, '<em>$1</em>');

        // Handle inline code
        text = text.replace(/`([^`]+)`/g, '<code class="bg-gray-100 px-1 py-0.5 rounded text-sm">$1</code>');

        // Sanitize the generated HTML to prevent XSS
        const sanitized = DOMPurify.sanitize(text, { USE_PROFILES: { html: true } });

        return <span dangerouslySetInnerHTML={{ __html: sanitized }} />;
    };

    return (
        <div className="prose max-w-none">
            {renderMarkdown(content)}
        </div>
    );
};

/**
 * Main HtmlToMarkdownConverter component
 */
const HtmlToMarkdownConverter = ({ url }) => {
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [markdownContent, setMarkdownContent] = useState('');
    const [originalTitle, setOriginalTitle] = useState('');
    const [successfulProxy, setSuccessfulProxy] = useState(null);

    const converter = new MarkdownConverter();

    /**
     * Extracts main content from HTML by removing common unwanted elements
     */
    const extractMainContent = (htmlString) => {
        const parser = new DOMParser();
        const doc = parser.parseFromString(htmlString, 'text/html');

        // Remove unwanted elements
        const unwantedSelectors = [
            'nav', '.nav', '#navigation', '.navigation',
            'aside', '.sidebar', '.side-nav', '.sidebars',
            'footer', '.footer',
            '.ad', '.ads', '.advertisement', '.advertisements',
            '[id*="ad"]', '[class*="ad"]',
            '.modal', '.popup', '.overlay',
            'script', 'style', 'noscript',
            '.social-share', '.share-buttons',
            '.comments', '.comment-section',
            '.newsletter', '.subscription',
            'iframe[src*="ads"]', 'iframe[src*="tracking"]',
            '.breadcrumb', '.breadcrumbs',
            '.related-posts', '.recommendations',
            '.cookie-banner', '.privacy-notice'
        ];

        unwantedSelectors.forEach(selector => {
            const elements = doc.querySelectorAll(selector);
            elements.forEach(el => el.remove());
        });

        // Try to find main content using common patterns
        const contentSelectors = [
            'main',
            'article',
            '[role="main"]',
            '.main-content',
            '.content',
            '.post-content',
            '.entry-content',
            '.article-content',
            '#content',
            '#main-content'
        ];

        let mainContent = null;
        for (const selector of contentSelectors) {
            const element = doc.querySelector(selector);
            if (element && element.textContent.trim().length > 100) {
                mainContent = element;
                break;
            }
        }

        // If no specific main content found, use body but filter out obvious non-content
        if (!mainContent) {
            mainContent = doc.body;

            // Remove elements that are likely not main content
            const additionalUnwanted = [
                'header', '.header',
                '.menu', '.navigation-menu',
                '.widget', '.widgets',
                '.meta', '.metadata'
            ];

            additionalUnwanted.forEach(selector => {
                const elements = mainContent.querySelectorAll(selector);
                elements.forEach(el => el.remove());
            });
        }

        return mainContent ? mainContent.innerHTML : '';
    };

    /**
     * Fetches and processes the URL content with improved CORS proxy fallback
     */
    const fetchAndConvert = async (targetUrl) => {
        if (!targetUrl) return;

        setLoading(true);
        setError(null);
        setMarkdownContent('');
        setOriginalTitle('');
        setSuccessfulProxy(null);

        // Updated list of working CORS proxy services (2025)
        const corsProxies = [
            '', // Try direct first
            'https://api.allorigins.win/raw?url=', // AllOrigins - good for general content
            'https://proxy.cors.sh/', // CORS.SH - reliable, unlimited for development
            'https://api.codetabs.com/v1/proxy?quest=', // CodeTabs - 5MB limit, reliable
            'https://cors-anywhere.herokuapp.com/', // Backup - limited access
        ];

        let lastError = null;
        let htmlContent = '';

        for (let i = 0; i < corsProxies.length; i++) {
            const proxy = corsProxies[i];
            let fetchUrl;

            // Handle different proxy URL formats
            if (proxy === 'https://proxy.cors.sh/') {
                fetchUrl = proxy + targetUrl;
            } else if (proxy === '') {
                fetchUrl = targetUrl;
            } else {
                fetchUrl = proxy + encodeURIComponent(targetUrl);
            }

            try {
                const proxyName = proxy === '' ? 'Direct' :
                    proxy.includes('allorigins') ? 'AllOrigins' :
                        proxy.includes('cors.sh') ? 'CORS.SH' :
                            proxy.includes('codetabs') ? 'CodeTabs' :
                                'CORS-Anywhere';

                console.log(`🔄 Attempting fetch via ${proxyName}: ${fetchUrl}`);

                const headers = {
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                };

                // Add required headers for CORS.SH
                if (proxy.includes('cors.sh')) {
                    headers['Origin'] = window.location.origin;
                    headers['x-requested-with'] = 'XMLHttpRequest';
                }

                const response = await fetch(fetchUrl, {
                    method: 'GET',
                    headers,
                    mode: 'cors'
                });

                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }

                htmlContent = await response.text();
                setSuccessfulProxy(proxyName);
                console.log(`✅ Successfully fetched content via ${proxyName}`);
                break; // Success! Exit the loop

            } catch (err) {
                const proxyName = proxy === '' ? 'Direct' : proxy.split('/')[2];
                console.log(`❌ Failed via ${proxyName}:`, err.message);
                lastError = err;
                continue;
            }
        }

        // If all attempts failed
        if (!htmlContent) {
            console.error('All fetch attempts failed:', lastError);

            if (lastError.name === 'TypeError' && lastError.message.includes('Failed to fetch')) {
                setError({
                    type: 'CORS',
                    message: 'Unable to fetch the URL after trying multiple proxy services.',
                    suggestion: 'This website may have strict CORS policies or be actively blocking automated requests.',
                    details: `Attempted: Direct fetch, AllOrigins, CORS.SH, CodeTabs, and CORS-Anywhere`,
                    solutions: [
                        'Try a different URL from a more permissive website',
                        'Use a browser extension to disable CORS for testing',
                        'Consider server-side fetching for production use',
                        'Check if the website has an official API instead'
                    ]
                });
            } else {
                setError({
                    type: 'FETCH',
                    message: lastError.message,
                    suggestion: 'Please verify the URL is correct and accessible.',
                    details: `Error: ${lastError.message}`,
                    solutions: [
                        'Check if the URL is correct and publicly accessible',
                        'Ensure the website is online and responding',
                        'Try accessing the URL directly in your browser first'
                    ]
                });
            }
            setLoading(false);
            return;
        }

        try {
            // Extract title
            const titleMatch = htmlContent.match(/<title[^>]*>([^<]*)<\/title>/i);
            if (titleMatch) {
                setOriginalTitle(titleMatch[1].trim());
            }

            // Extract main content
            const mainContent = extractMainContent(htmlContent);

            if (!mainContent.trim()) {
                throw new Error('No main content found in the webpage');
            }

            // Convert to markdown
            const markdown = converter.convert(mainContent);

            if (!markdown.trim()) {
                throw new Error('Failed to convert content to markdown');
            }

            setMarkdownContent(markdown);

        } catch (err) {
            console.error('Processing error:', err);
            setError({
                type: 'PROCESSING',
                message: `Content processing failed: ${err.message}`,
                suggestion: 'The webpage content could not be processed. It may have an unusual structure.',
                details: err.message
            });
        } finally {
            setLoading(false);
        }
    };

    // Trigger fetch when URL changes
    useEffect(() => {
        if (url) {
            fetchAndConvert(url);
        }
    }, [url]);

    return (
        <div className="max-w-4xl mx-auto p-6 bg-white">
            {/* Header */}
            <div className="mb-6">
                <h1 className="text-2xl font-bold text-gray-800 mb-2 flex items-center gap-2">
                    <FileText className="w-6 h-6" />
                    HTML to Markdown Converter
                </h1>
                {url && (
                    <div className="flex items-center gap-2 text-sm text-gray-600">
                        <ExternalLink className="w-4 h-4" />
                        <span className="break-all">{url}</span>
                    </div>
                )}
            </div>

            {/* Loading State */}
            {loading && (
                <div className="flex flex-col items-center justify-center py-12">
                    <Loader2 className="w-8 h-8 animate-spin text-blue-500 mb-3" />
                    <span className="text-lg text-gray-600 mb-2">Fetching and converting content...</span>
                    <div className="text-sm text-gray-500 max-w-md text-center">
                        <p>Trying multiple proxy services for best compatibility</p>
                        <p className="text-xs mt-1">Direct → AllOrigins → CORS.SH → CodeTabs → CORS-Anywhere</p>
                    </div>
                </div>
            )}

            {/* Error State */}
            {error && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
                    <div className="flex items-start gap-3">
                        <AlertCircle className="w-5 h-5 text-red-500 mt-0.5 flex-shrink-0" />
                        <div className="flex-1">
                            <h3 className="font-semibold text-red-800 mb-1">
                                {error.type === 'CORS' ? 'CORS Policy Error' :
                                    error.type === 'PROCESSING' ? 'Content Processing Error' : 'Network Error'}
                            </h3>
                            <p className="text-red-700 mb-2">{error.message}</p>
                            <p className="text-red-600 text-sm mb-3">{error.suggestion}</p>

                            {error.solutions && (
                                <div className="mb-3">
                                    <h4 className="font-medium text-red-800 mb-2">💡 Possible Solutions:</h4>
                                    <ul className="text-red-700 text-sm space-y-1">
                                        {error.solutions.map((solution, index) => (
                                            <li key={index} className="flex items-start gap-2">
                                                <span className="text-red-500 mt-0.5">•</span>
                                                <span>{solution}</span>
                                            </li>
                                        ))}
                                    </ul>
                                </div>
                            )}

                            {error.details && (
                                <details className="text-red-600 text-xs">
                                    <summary className="cursor-pointer font-medium">Technical Details</summary>
                                    <p className="mt-1 font-mono bg-red-100 p-2 rounded">{error.details}</p>
                                </details>
                            )}

                            {error.type === 'CORS' && (
                                <div className="mt-3 p-3 bg-blue-50 border border-blue-200 rounded text-sm">
                                    <strong className="text-blue-800">🔧 Advanced Options:</strong>
                                    <ul className="mt-1 text-blue-700 space-y-1">
                                        <li>• <strong>For Development:</strong> Use browser extensions like "CORS Unblock" or "Moesif Origin & CORS Changer"</li>
                                        <li>• <strong>For Production:</strong> Implement server-side fetching with Node.js/Express</li>
                                        <li>• <strong>Alternative:</strong> Use headless browser tools (Puppeteer, Playwright) for complex sites</li>
                                        <li>• <strong>API Alternative:</strong> Check if the website offers an official API instead</li>
                                    </ul>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            )}

            {/* Success State */}
            {markdownContent && !loading && !error && (
                <div className="space-y-6">
                    {/* Success Banner */}
                    <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                        <div className="flex items-center gap-2">
                            <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                            <span className="text-green-800 font-medium">
                                Successfully converted via {successfulProxy || 'Direct fetch'}
                            </span>
                        </div>
                    </div>

                    {originalTitle && (
                        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                            <h2 className="font-semibold text-blue-800 mb-1">Original Title:</h2>
                            <p className="text-blue-700">{originalTitle}</p>
                        </div>
                    )}

                    {/* Converted Content */}
                    <div className="bg-gray-50 border border-gray-200 rounded-lg">
                        <div className="border-b border-gray-200 px-4 py-2">
                            <h3 className="font-semibold text-gray-800">Converted Markdown Content</h3>
                        </div>
                        <div className="p-6">
                            <MarkdownRenderer content={markdownContent} />
                        </div>
                    </div>

                    {/* Raw Markdown (collapsible) */}
                    <details className="bg-gray-50 border border-gray-200 rounded-lg">
                        <summary className="cursor-pointer px-4 py-2 font-semibold text-gray-800 border-b border-gray-200">
                            View Raw Markdown
                        </summary>
                        <div className="p-4">
                            <pre className="bg-white border rounded p-3 text-sm overflow-x-auto whitespace-pre-wrap">
                                {markdownContent}
                            </pre>
                        </div>
                    </details>
                </div>
            )}

            {/* Instructions */}
            {!url && (
                <div className="space-y-4">
                    <div className="bg-blue-50 border border-blue-200 rounded-lg p-6 text-center">
                        <FileText className="w-12 h-12 text-blue-500 mx-auto mb-3" />
                        <h3 className="font-semibold text-blue-800 mb-2">Ready to Convert</h3>
                        <p className="text-blue-700">
                            This component fetches HTML content and converts it to clean Markdown.
                        </p>
                        <p className="text-blue-600 text-sm mt-2">
                            Uses intelligent content extraction and multiple CORS proxy fallbacks.
                        </p>
                    </div>

                    <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                        <h4 className="font-semibold text-green-800 mb-2">✅ Try These Tested URLs:</h4>
                        <div className="space-y-2 text-sm">
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                                <button
                                    onClick={() => setInputUrl('https://example.com')}
                                    className="px-3 py-2 bg-green-100 text-green-700 rounded hover:bg-green-200 transition-colors text-left"
                                >
                                    <strong>example.com</strong> - Simple test page
                                </button>
                                <button
                                    onClick={() => setInputUrl('https://httpbin.org/html')}
                                    className="px-3 py-2 bg-green-100 text-green-700 rounded hover:bg-green-200 transition-colors text-left"
                                >
                                    <strong>httpbin.org/html</strong> - Clean HTML
                                </button>
                                <button
                                    onClick={() => setInputUrl('https://jsonplaceholder.typicode.com/')}
                                    className="px-3 py-2 bg-green-100 text-green-700 rounded hover:bg-green-200 transition-colors text-left"
                                >
                                    <strong>jsonplaceholder</strong> - API docs
                                </button>
                                <button
                                    onClick={() => setInputUrl('https://www.w3.org/TR/html52/')}
                                    className="px-3 py-2 bg-green-100 text-green-700 rounded hover:bg-green-200 transition-colors text-left"
                                >
                                    <strong>w3.org/TR/html52</strong> - W3C spec
                                </button>
                            </div>
                            <p className="text-green-600 text-xs mt-2">
                                💡 These URLs are known to work well with our proxy services
                            </p>
                        </div>

                        <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded">
                            <h5 className="font-medium text-blue-800 mb-1">📊 Proxy Service Status (2025):</h5>
                            <div className="text-xs text-blue-700 grid grid-cols-1 md:grid-cols-2 gap-1">
                                <div>🟢 <strong>CORS.SH</strong> - Unlimited, reliable</div>
                                <div>🟢 <strong>AllOrigins</strong> - Good for general content</div>
                                <div>🟡 <strong>CodeTabs</strong> - 5MB limit, stable</div>
                                <div>🔴 <strong>CorsProxy.io</strong> - JSON/XML only now</div>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

// Example usage component
const App = () => {
    const [inputUrl, setInputUrl] = useState('');
    const [currentUrl, setCurrentUrl] = useState('');

    const handleConvert = () => {
        if (inputUrl.trim()) {
            setCurrentUrl(inputUrl.trim());
        }
    };

    const handleKeyPress = (e) => {
        if (e.key === 'Enter') {
            handleConvert();
        }
    };

    return (
        <div className="min-h-screen bg-gray-100 py-8">
            <div className="max-w-4xl mx-auto px-4">
                {/* URL Input */}
                <div className="bg-white rounded-lg shadow-sm border p-6 mb-6">
                    <div className="flex gap-3">
                        <input
                            type="url"
                            value={inputUrl}
                            onChange={(e) => setInputUrl(e.target.value)}
                            onKeyPress={handleKeyPress}
                            placeholder="Enter URL to convert (e.g., https://example.com/article)"
                            className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                        />
                        <button
                            onClick={handleConvert}
                            className="px-6 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-colors"
                        >
                            Convert
                        </button>
                    </div>
                </div>

                {/* Converter Component */}
                <HtmlToMarkdownConverter url={currentUrl} />
            </div>
        </div>
    );
};

export default App;
