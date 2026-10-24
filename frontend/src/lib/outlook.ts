import type { Outlook } from '@/lib/api';

const DIRECTION_LABEL: Record<string, string> = { emerging: 'Rising', declining: 'Falling', stable: 'Stable' };

export type Chip = { label: string; tone: 'rising' | 'falling' | 'stable' | 'none'; unconfirmed: boolean };

export function outlookChip(outlook: Outlook, validated: boolean): Chip {
  const hasForecast = (outlook.forecast_status === 'ok' || outlook.forecast_status === 'preview') && outlook.direction;
  if (!hasForecast || !outlook.direction) return { label: 'No outlook', tone: 'none', unconfirmed: false };
  const tone = outlook.direction === 'emerging' ? 'rising' : outlook.direction === 'declining' ? 'falling' : 'stable';
  const unconfirmed = outlook.confidence === 'low' || outlook.signal_quality === 'noise_like';
  const label = `${DIRECTION_LABEL[outlook.direction] ?? outlook.direction}${validated ? '' : ' (unvalidated)'}`;
  return { label, tone, unconfirmed };
}

export function percent(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  const shown = value * 100;
  return `${shown > 0 ? '+' : ''}${shown.toFixed(digits)}%`;
}

export function fromLog(value: number | null | undefined): number | null {
  return value === null || value === undefined ? null : Math.exp(value) - 1;
}

export function readableId(id: string): string {
  const name = id.split(':').pop() ?? id;
  return name.replace(/_/g, ' ');
}
