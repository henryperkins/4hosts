# Research Progress Remediation Plan

## 🎯 Objective
Transform the research progress display from a basic status indicator into a comprehensive, real-time research dashboard that surfaces all backend intelligence to users.

---

## 📋 Phase 1: Critical Fixes (1-2 days)
*Immediate impact, minimal code changes*

### 1.1 Handle Missing Events
```typescript
// ResearchProgress.tsx:261
case 'system.notification':
  // Handle MCP tool events and other notifications
  if (data.message?.includes('mcp.tool')) {
    statusUpdate = {
      message: `🔧 ${data.message}`,
      // Add to a new MCP tools counter
    }
    setStats(prev => ({ ...prev, mcpToolsUsed: (prev.mcpToolsUsed || 0) + 1 }))
  } else {
    statusUpdate = { message: data.message }
  }
  break

case 'rate_limit.warning':
  statusUpdate = {
    message: `⚠️ Rate limit warning: ${data.message}`
  }
  // Show persistent warning banner
  setRateLimitWarning(data)
  break
```

### 1.2 Fix Default Filtering
```typescript
// Line 370 - Make filtering smarter
const isNoisy = (msg?: string) => {
  const m = (msg || '').toLowerCase()
  // Only hide truly repetitive messages
  return m.includes('heartbeat') || m.includes('still processing')
}

// OR just default to verbose
const [showVerbose, setShowVerbose] = useState<boolean>(true) // was false
```

### 1.3 Add Deduplication to Stats Grid
```typescript
// Line 81 - Extend ResearchStats
interface ResearchStats {
  // ... existing fields
  duplicatesRemoved: number
  mcpToolsUsed: number
  apisQueried: Set<string>
}

// Line 256 - Update dedup handler
case 'deduplication.progress':
  statusUpdate = {
    message: `Removed ${data.removed} duplicate results`
  }
  setStats(prev => ({
    ...prev,
    duplicatesRemoved: (prev.duplicatesRemoved || 0) + (data.removed || 0)
  }))
  break
```

### 1.4 Persist Key Metrics
```typescript
// Add new statistics cards (Line 519+)
<div className="bg-surface-subtle rounded-lg p-3">
  <div className="text-2xl font-bold text-warning">{stats.duplicatesRemoved || 0}</div>
  <div className="text-xs text-text-muted">Duplicates Removed</div>
</div>
<div className="bg-surface-subtle rounded-lg p-3">
  <div className="text-2xl font-bold text-info">{stats.apisQueried.size || 0}</div>
  <div className="text-xs text-text-muted">APIs Used</div>
</div>
```

---

## 📊 Phase 2: Enhanced Event Visibility (2-3 days)

### 2.1 Categorized Event Display
```typescript
// Group messages by type
interface CategorizedUpdates {
  search: ProgressUpdate[]
  sources: ProgressUpdate[]
  analysis: ProgressUpdate[]
  system: ProgressUpdate[]
  errors: ProgressUpdate[]
}

// Add tabs for different event categories
<Tabs defaultValue="all">
  <TabsList>
    <TabsTrigger value="all">All</TabsTrigger>
    <TabsTrigger value="search">Search ({categorized.search.length})</TabsTrigger>
    <TabsTrigger value="sources">Sources ({categorized.sources.length})</TabsTrigger>
    <TabsTrigger value="analysis">Analysis ({categorized.analysis.length})</TabsTrigger>
    <TabsTrigger value="errors" className={errors.length > 0 ? 'text-error' : ''}>
      Errors ({categorized.errors.length})
    </TabsTrigger>
  </TabsList>
  <TabsContent value="all">{/* Existing log */}</TabsContent>
  <TabsContent value="search">{/* Search events only */}</TabsContent>
  {/* ... other tabs */}
</Tabs>
```

### 2.2 Event Priority System
```typescript
// Define event priorities
enum EventPriority {
  CRITICAL = 'critical',  // Errors, failures
  HIGH = 'high',         // Phase changes, completion
  MEDIUM = 'medium',     // Source found, dedup
  LOW = 'low'           // Routine progress
}

// Style messages by priority
const getMessageStyle = (priority: EventPriority) => {
  switch(priority) {
    case EventPriority.CRITICAL:
      return 'border-l-4 border-error bg-error/5'
    case EventPriority.HIGH:
      return 'border-l-4 border-primary bg-primary/5'
    default:
      return ''
  }
}
```

### 2.3 Persistent Event Summary
```typescript
// Add summary section above log
<div className="mb-4 p-3 bg-surface-subtle rounded-lg">
  <h4 className="text-sm font-medium mb-2">Session Summary</h4>
  <div className="grid grid-cols-2 gap-2 text-xs">
    <div>✅ {stats.searchesCompleted} searches completed</div>
    <div>📊 {stats.sourcesAnalyzed} sources analyzed</div>
    <div>🔍 {stats.highQualitySources} high-quality sources</div>
    <div>🗑️ {stats.duplicatesRemoved} duplicates removed</div>
    <div>🔧 {stats.mcpToolsUsed} tools executed</div>
    <div>⚡ {stats.apisQueried.size} APIs queried</div>
  </div>
</div>
```

---

## 🎨 Phase 3: UI/UX Improvements (3-4 days)

### 3.1 Enhanced Progress Visualization
```typescript
// Multi-layer progress bar showing sub-phases
<div className="space-y-1">
  <ProgressBar
    value={overallProgress}
    label="Overall"
    variant="primary"
  />
  {currentPhase === 'search' && (
    <ProgressBar
      value={(stats.searchesCompleted / stats.totalSearches) * 100}
      label={`Search ${stats.searchesCompleted}/${stats.totalSearches}`}
      variant="secondary"
      size="sm"
    />
  )}
  {currentPhase === 'analysis' && (
    <ProgressBar
      value={(stats.sourcesAnalyzed / analysisTotal) * 100}
      label={`Analyzing ${stats.sourcesAnalyzed}/${analysisTotal}`}
      variant="secondary"
      size="sm"
    />
  )}
</div>
```

### 3.2 API Status Indicators
```typescript
// Show which APIs are active
<div className="flex gap-2 mb-2">
  {Array.from(stats.apisQueried).map(api => (
    <Badge
      key={api}
      variant={activeApis.has(api) ? 'default' : 'subtle'}
      size="sm"
      className={activeApis.has(api) ? 'animate-pulse' : ''}
    >
      {api}
    </Badge>
  ))}
</div>
```

### 3.3 Improved Mobile Layout
```typescript
// Collapsible sections for mobile
const [sectionsCollapsed, setSectionsCollapsed] = useState({
  stats: false,
  timeline: window.innerWidth < 640,
  messages: false,
  sources: true
})

// Swipeable tabs for mobile
<SwipeableViews index={activeTab} onChangeIndex={setActiveTab}>
  <div>Stats</div>
  <div>Timeline</div>
  <div>Messages</div>
  <div>Sources</div>
</SwipeableViews>
```

### 3.4 Smart Filtering Options
```typescript
// Replace binary verbose/concise with smart filters
interface FilterOptions {
  showSearch: boolean
  showSources: boolean
  showAnalysis: boolean
  showSystem: boolean
  minPriority: EventPriority
}

// Filter UI
<Popover>
  <PopoverTrigger>
    <Button variant="ghost" size="sm">
      Filter <FiFilter />
    </Button>
  </PopoverTrigger>
  <PopoverContent>
    <div className="space-y-2">
      <Checkbox checked={filters.showSearch} onChange={...}>
        Search Events
      </Checkbox>
      {/* ... other filters */}
    </div>
  </PopoverContent>
</Popover>
```

---

## 🚀 Phase 4: Advanced Features (1 week)

### 4.1 Event Timeline Visualization
```typescript
// D3.js or Recharts timeline
<Timeline
  events={updates}
  phases={researchPhases}
  currentTime={elapsedSec}
  onEventClick={(event) => setSelectedEvent(event)}
/>
```

### 4.2 Real-time Metrics Dashboard
```typescript
// Live updating charts
<div className="grid grid-cols-2 gap-4">
  <LineChart
    data={searchRateHistory}
    title="Search Rate"
    yAxis="queries/min"
  />
  <PieChart
    data={sourceQualityDistribution}
    title="Source Quality"
  />
</div>
```

### 4.3 Expandable Event Details
```typescript
// Click to expand events
{updates.map(update => (
  <Collapsible key={update.id}>
    <CollapsibleTrigger className="w-full">
      <div className="flex justify-between">
        <span>{update.message}</span>
        <FiChevronDown />
      </div>
    </CollapsibleTrigger>
    <CollapsibleContent>
      <pre className="text-xs bg-surface-subtle p-2 rounded">
        {JSON.stringify(update.data, null, 2)}
      </pre>
    </CollapsibleContent>
  </Collapsible>
))}
```

### 4.4 Export and History
```typescript
// Export progress log
<Button onClick={() => exportProgressLog(updates)}>
  Export Log <FiDownload />
</Button>

// View previous research progress
<Button onClick={() => loadHistoricalProgress(researchId)}>
  View History <FiClock />
</Button>
```

---

## 📅 Implementation Timeline

| Phase | Duration | Impact | Effort | Priority |
|-------|----------|--------|--------|----------|
| **Phase 1** | 1-2 days | High | Low | **CRITICAL** |
| 1.1 Handle Events | 2 hours | High | Low | 1 |
| 1.2 Fix Filtering | 1 hour | High | Low | 2 |
| 1.3 Add Stats | 2 hours | Medium | Low | 3 |
| 1.4 Persist Metrics | 2 hours | Medium | Low | 4 |
| **Phase 2** | 2-3 days | High | Medium | **HIGH** |
| 2.1 Categorize | 4 hours | High | Medium | 5 |
| 2.2 Priority | 3 hours | Medium | Low | 6 |
| 2.3 Summary | 3 hours | High | Low | 7 |
| **Phase 3** | 3-4 days | Medium | Medium | **MEDIUM** |
| 3.1 Progress Viz | 4 hours | Medium | Medium | 8 |
| 3.2 API Status | 3 hours | Low | Low | 9 |
| 3.3 Mobile | 6 hours | High | High | 10 |
| 3.4 Smart Filter | 4 hours | Medium | Medium | 11 |
| **Phase 4** | 1 week | Low | High | **LOW** |
| 4.1 Timeline | 8 hours | Low | High | 12 |
| 4.2 Dashboard | 8 hours | Low | High | 13 |
| 4.3 Details | 4 hours | Low | Medium | 14 |
| 4.4 Export | 4 hours | Low | Medium | 15 |

---

## ✅ Success Metrics

### After Phase 1:
- ✅ All backend events visible
- ✅ Deduplication impact shown
- ✅ MCP tool usage tracked
- ✅ No information loss

### After Phase 2:
- ✅ Events organized by category
- ✅ Important events highlighted
- ✅ Persistent summary available
- ✅ Error visibility improved

### After Phase 3:
- ✅ Mobile experience optimized
- ✅ Granular progress tracking
- ✅ API status visible
- ✅ Flexible filtering

### After Phase 4:
- ✅ Historical analysis possible
- ✅ Visual timeline available
- ✅ Exportable metrics
- ✅ Professional dashboard

---

## 🔧 Testing Strategy

1. **Unit Tests**: Event handler coverage
2. **Integration Tests**: WebSocket message flow
3. **E2E Tests**: Full research progress tracking
4. **Performance Tests**: 1000+ events handling
5. **Mobile Tests**: Responsive design validation

---

## 📝 Migration Notes

- Phase 1 is **backward compatible**
- No backend changes required
- Existing research sessions unaffected
- Gradual rollout possible

---

## 🎯 Current State Analysis

### Backend Implementation Status (85% Complete)
- ✅ **Deduplication Reporting**: Fully implemented at `research_orchestrator.py:1257`
- ✅ **Context Engineering Granularity**: Reports items_done/items_total for each WSCI layer
- ✅ **MCP Tool Broadcasting**: Sends `MCP_TOOL_EXECUTING` and `MCP_TOOL_COMPLETED` events
- ✅ **Background LLM Progress**: Comprehensive heartbeat system every 5 seconds
- ✅ **Search API Granularity**: Provider-specific search start/completion events
- ⚠️ **Rate Limit Warnings**: Circuit breaker exists but doesn't broadcast
- ⚠️ **Deep Research Granularity**: Tool-by-tool progress not itemized

### Frontend Rendering Issues
- **70% of events hidden** by default filtering
- **MCP events lost** due to missing `system.notification` handler
- **Ephemeral display** causes information loss
- **Poor mobile experience** with limited viewport
- **No event categorization** or priority system
- **Missing persistence** for key metrics like deduplication

### User Impact
- Users see simplified progress missing research process richness
- Critical information scrolls away permanently
- No visibility into external tool usage
- Deduplication benefits not apparent
- API-specific progress hidden

---

## 📊 Expected Outcomes

### Immediate (Phase 1)
- 100% event visibility
- Persistent metrics display
- Reduced user confusion
- Better progress understanding

### Short-term (Phase 2-3)
- Organized information hierarchy
- Mobile-optimized experience
- Customizable views
- Professional appearance

### Long-term (Phase 4)
- Data-driven insights
- Historical comparisons
- Export capabilities
- Enterprise-ready dashboard