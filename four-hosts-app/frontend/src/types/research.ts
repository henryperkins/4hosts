// Research API contracts mirrored in TypeScript for FE safety

export type ResearchStatus = 'queued' | 'processing' | 'in_progress' | 'completed' | 'failed' | 'cancelled';
export type Paradigm = 'dolores' | 'teddy' | 'bernard' | 'maeve';

export interface AnswerSection {
  title: string;
  paradigm: Paradigm;
  content: string;
  confidence: number;
  sources_count: number;
  citations: any[];
  key_insights?: string[];
  // Allow backend to add extras without breaking FE
  [key: string]: any;
}

export interface AnswerPayload {
  summary: string;
  sections: AnswerSection[];
  action_items: any[];
  citations: any[];
  metadata: Record<string, any>;
}

export interface ParadigmSummary {
  paradigm: Paradigm;
  confidence: number;
  approach?: string;
  focus?: string;
}

export interface ParadigmAnalysis {
  primary: ParadigmSummary;
  secondary?: ParadigmSummary;
}

export interface SourceResult {
  title: string;
  url: string;
  snippet: string;
  domain: string;
  credibility_score: number;
  published_date?: string;
  source_type?: string;
  source_category?: string;
  credibility_explanation?: string;
}

export interface ResearchFinalResult {
  research_id: string;
  query: string;
  status: ResearchStatus;
  paradigm_analysis: ParadigmAnalysis;
  answer: AnswerPayload;
  integrated_synthesis?: Record<string, any> | null;
  mesh_synthesis?: Record<string, any> | null;
  sources: SourceResult[];
  metadata: Record<string, any>;
  cost_info?: Record<string, any>;
  export_formats?: Record<string, string>;
}

