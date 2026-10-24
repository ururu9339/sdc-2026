export const API_BASE = `${(import.meta.env.VITE_API_URL ?? '').replace(/\/$/, '')}/api/v1`;

export type Outlook = {
  growth: number | null;
  forecast_status: string | null;
  direction: string | null;
  confidence: string | null;
  signal_quality: string | null;
  shares_signal_with?: string | null;
};
export type SkillRef = { skill_id: string; name: string };
export type JobFit = Outlook & { job_id: string; name: string; coverage: number; matched: SkillRef[]; missing: SkillRef[] };
export type SkillGap = Outlook & { skill_id: string; name: string; score: number; for_jobs: string[]; stage: string; why: string };
export type ResolvedSkill = { input: string; skill_id: string | null; name: string | null; status: string; method: string };
export type Report = {
  engine: string;
  feed: string;
  validated: boolean;
  banner: string | null;
  forecast_window: string | null;
  stale: boolean | null;
  status: string;
  message: string | null;
  resolved_skills: ResolvedSkill[];
  unresolved: string[];
  best_fit_jobs: JobFit[];
  skill_gaps: SkillGap[];
  learning_path: { note: string; stages: { title: string; skills: SkillRef[] }[] };
  market_snapshot: { entity_id: string; name: string; percent_change: number | null }[];
  caveats: string[];
  warnings: string[];
};
export type ForecastRecord = {
  entity_id: string;
  kind: string;
  status: string;
  p50: number | null;
  p10: number | null;
  p90: number | null;
  direction: string | null;
  confidence: string | null;
  signal_quality: string | null;
  shares_signal_with: string | null;
};
export type ForecastList = {
  feed: string;
  validated: boolean;
  banner: string | null;
  forecast_window: string | null;
  stale: boolean | null;
  total: number;
  forecasts: ForecastRecord[];
};
type TaxonomyList = { items: { id: string; name: string }[] };

export class ApiError extends Error {
  constructor(public status: number, public code: string, message: string, public requestId: string | null) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, { ...init, headers: { Accept: 'application/json', ...init?.headers } });
  } catch {
    throw new ApiError(0, 'unreachable', 'The service could not be reached.', null);
  }
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    const error = body?.error ?? {};
    throw new ApiError(response.status, error.code ?? 'error', error.message ?? response.statusText, error.request_id ?? null);
  }
  return response.json() as Promise<T>;
}

export function reportFromProfile(skills: string[], text: string): Promise<Report> {
  return request<Report>('/report', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ skills, text: text || null, horizon_months: 3 }),
  });
}

export async function reportFromCv(file: File): Promise<Report> {
  const form = new FormData();
  form.append('file', file, file.name);
  const created = await request<{ candidate: { id: string } }>('/candidates/pdf', { method: 'POST', body: form });
  return request<Report>(`/report/candidate/${encodeURIComponent(created.candidate.id)}`);
}

export function forecasts(direction: 'emerging' | 'declining'): Promise<ForecastList> {
  return request<ForecastList>(`/forecast?direction=${direction}&limit=50`);
}

export async function entityNames(): Promise<Record<string, string>> {
  const [skills, jobs] = await Promise.all([
    request<TaxonomyList>('/taxonomy/skills?limit=500'),
    request<TaxonomyList>('/taxonomy/jobs?limit=500'),
  ]);
  return Object.fromEntries([...skills.items, ...jobs.items].map((item) => [item.id, item.name]));
}
