export type HostParadigm = 'dolores' | 'teddy' | 'bernard' | 'maeve';

export type Paradigm = HostParadigm;

export interface ParadigmInfo {
  name: HostParadigm;
  title: string;
  description: string;
  icon: string;
  color: string;
}

export const PARADIGM_INFO: Record<HostParadigm, ParadigmInfo> = {
  dolores: {
    name: 'dolores',
    title: 'Revolutionary',
    description: 'Investigative research that exposes systemic issues',
    icon: '✊',
    color: 'red'
  },
  teddy: {
    name: 'teddy',
    title: 'Devotion',
    description: 'Supportive and community-focused empathetic research',
    icon: '❤️',
    color: 'blue'
  },
  bernard: {
    name: 'bernard',
    title: 'Analytical',
    description: 'Data-driven empirical research with academic rigor',
    icon: '📊',
    color: 'gray'
  },
  maeve: {
    name: 'maeve',
    title: 'Strategic',
    description: 'Business intelligence with actionable strategies',
    icon: '♟️',
    color: 'purple'
  }
};