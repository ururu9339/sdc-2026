import { useQuery } from '@tanstack/react-query';
import { entityNames, forecasts, type ForecastList, type ForecastRecord } from '@/lib/api';
import { useLanguage } from '@/lib/language';
import { fromLog, percent } from '@/lib/outlook';
import { OutlookChip } from '@/components/report-view';

function grouped(list: ForecastRecord[]) {
  const canonical = list.filter((r) => !r.shares_signal_with || !list.some((o) => o.entity_id === r.shares_signal_with));
  return canonical.map((record) => ({ record, copies: list.filter((o) => o.shares_signal_with === record.entity_id) }));
}

function SignalColumn({ title, data, names }: { title: string; data: ForecastList; names: Record<string, string> }) {
  const { t } = useLanguage();
  const rows = grouped(data.forecasts);
  return (
    <div className="signal-column">
      <h3>{t(title)}</h3>
      {rows.length === 0 ? <p className="report-muted">{t('No matches for these filters.')}</p> : (
        <ol>
          {rows.map(({ record, copies }) => (
            <li key={record.entity_id} data-testid={`signal-${record.entity_id}`}>
              <div className="job-head">
                <strong>{names[record.entity_id] ?? record.entity_id}</strong>
                <span className="report-muted">{t(record.kind === 'job' ? 'Job' : 'Skill')}</span>
              </div>
              <OutlookChip
                outlook={{ ...record, growth: fromLog(record.p50), forecast_status: record.status }}
                validated={data.validated} window={data.forecast_window} stale={data.stale} />
              <p className="report-muted">{t('Range')}: {percent(fromLog(record.p10), 0)} {t('to')} {percent(fromLog(record.p90), 0)}</p>
              {copies.length > 0 && (
                <p className="report-muted">{t('Same signal as')}: {copies.map((c) => names[c.entity_id] ?? c.entity_id).join(', ')}</p>
              )}
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}

export function MarketSignals() {
  const { t } = useLanguage();
  const rising = useQuery({ queryKey: ['forecasts', 'emerging'], queryFn: () => forecasts('emerging') });
  const falling = useQuery({ queryKey: ['forecasts', 'declining'], queryFn: () => forecasts('declining') });
  const names = useQuery({ queryKey: ['entity-names'], queryFn: entityNames, staleTime: Infinity });
  const banner = rising.data?.banner ?? falling.data?.banner;

  return (
    <section className="market-signals" aria-labelledby="market-signals-title" data-testid="section-market-signals">
      <div className="detail-eyebrow"><span>{t('Market signals')}</span><span>{t('Skills and jobs whose share of postings is moving')}</span></div>
      <h2 id="market-signals-title">{t('Emerging and declining')}</h2>
      {banner && <p className="report-banner" role="status">{banner}</p>}
      {(rising.isError || falling.isError) && <p className="profile-error" role="alert">{t('Market signals are unavailable right now. Please try again later.')}</p>}
      {(rising.isPending || falling.isPending) && <p>{t('Loading market signals…')}</p>}
      {rising.data && falling.data && (
        <div className="signal-columns">
          <SignalColumn title="Emerging" data={rising.data} names={names.data ?? {}} />
          <SignalColumn title="Declining" data={falling.data} names={names.data ?? {}} />
        </div>
      )}
    </section>
  );
}
