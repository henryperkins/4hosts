import React from 'react'

export const TypographyExample: React.FC = () => {
  return (
    <div className="max-w-4xl mx-auto p-8">
      <h1 className="text-3xl font-bold text-text mb-8">Typography Plugin Example</h1>
      
      {/* Basic prose example */}
      <div className="prose prose-lg max-w-none mb-8">
        <h2>Using the Tailwind CSS Typography Plugin</h2>
        <p>
          The <strong>@tailwindcss/typography</strong> plugin provides beautiful typographic defaults for any vanilla HTML content. 
          It's perfect for rendering markdown content, blog posts, documentation, or any long-form content.
        </p>
        <h3>Key Features</h3>
        <ul>
          <li>Beautiful typographic defaults</li>
          <li>Dark mode support with <code>prose-invert</code></li>
          <li>Multiple size modifiers</li>
          <li>Customizable through Tailwind config</li>
        </ul>
        <blockquote>
          "Typography is the craft of endowing human language with a durable visual form."
        </blockquote>
      </div>

      {/* Dark mode example */}
      <div className="bg-surface-muted dark:bg-surface-subtle p-6 rounded-lg mb-8">
        <h3 className="text-xl font-semibold text-text mb-4">Dark Mode Support</h3>
        <div className="prose dark:prose-invert">
          <p>
            Use <code>dark:prose-invert</code> to automatically adjust colors for dark mode.
            All typography elements will adapt to your theme.
          </p>
        </div>
      </div>

      {/* Size variants */}
      <div className="space-y-6">
        <div className="prose prose-sm">
          <h4>Small Prose (prose-sm)</h4>
          <p>Perfect for sidebars, captions, or secondary content.</p>
        </div>
        
        <div className="prose">
          <h4>Default Prose</h4>
          <p>The standard size for most content.</p>
        </div>
        
        <div className="prose prose-lg">
          <h4>Large Prose (prose-lg)</h4>
          <p>Great for hero sections or featured content.</p>
        </div>
      </div>

      {/* Custom theme colors */}
      <div className="mt-8 p-6 bg-primary/5 rounded-lg">
        <div className="prose prose-primary">
          <h3>Theme Integration</h3>
          <p>
            The typography plugin has been configured to use your custom theme colors.
            Links use <a href="#">your primary color</a>, and all text respects your theme variables.
          </p>
        </div>
      </div>
    </div>
  )
}