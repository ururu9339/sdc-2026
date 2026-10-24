import { useEffect, useMemo, useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { useMutation } from '@tanstack/react-query';
import { z } from 'zod';
import { ArrowRight } from 'lucide-react';
import { Form, FormControl, FormDescription, FormField, FormItem, FormLabel, FormMessage } from '@/components/ui/form';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { useLanguage } from '@/lib/language';
import { CvUpload } from '@/components/cv-upload';
import { ReportView } from '@/components/report-view';
import { ApiError, reportFromCv, reportFromProfile, type Report } from '@/lib/api';

type ProfileValues = { name: string; skills: string; certificates: string; degree: string; age: string };
const defaultValues: ProfileValues = { name: '', skills: '', certificates: '', degree: '', age: '' };

function makeSchema(t: (s: string) => string) {
  const text = (msg: string) => z.string().refine((v) => v.trim().length > 0, { message: t(msg) });
  return z.object({
    name: text('Please enter your name.'),
    skills: text('Please describe your skills.'),
    certificates: text('Please list your certificates, or enter None.'),
    degree: text('Please enter your university degree, or enter None.'),
    age: z.string()
      .refine((v) => v.trim().length > 0, { message: t('Please enter your age.') })
      .refine((v) => v.trim() === '' || /^\d+$/.test(v.trim()), { message: t('Age must be a whole number.') })
      .refine((v) => !/^\d+$/.test(v.trim()) || (Number(v) >= 1 && Number(v) <= 120), { message: t('Age must be between 1 and 120.') }),
  });
}

const NONE = /^(none|tiada|n\/?a|-)$/i;

function profileRequest(values: ProfileValues): { skills: string[]; text: string } {
  const skills = values.skills.split(/[,;\n•·]+/).map((s) => s.trim())
    .filter((s) => s && s.length <= 200 && s.split(/\s+/).length <= 6).slice(0, 100);
  const text = [values.skills, values.certificates, values.degree].map((s) => s.trim())
    .filter((s) => s && !NONE.test(s)).join('.\n').slice(0, 20000);
  return { skills, text };
}

function errorMessage(error: unknown, fromCv: boolean): string {
  if (!(error instanceof ApiError)) return 'Something went wrong. Please try again.';
  if (error.status === 0) return 'The service could not be reached. Please check your connection and try again.';
  if (fromCv && (error.status === 503 || error.status === 422)) return 'We could not read this PDF right now. Please type your skills in the form instead.';
  if (error.status === 413) return 'Please choose a PDF smaller than 10 MB.';
  if (error.status === 415) return 'Please choose a PDF file.';
  if (error.status === 422) return 'Please shorten your skills list and try again.';
  return 'Something went wrong. Please try again.';
}

export function ProfileForm() {
  const { t } = useLanguage();
  const schema = useMemo(() => makeSchema(t), [t]);
  const [cv, setCv] = useState<File | null>(null);
  const [report, setReport] = useState<Report | null>(null);
  const form = useForm<ProfileValues>({ resolver: zodResolver(schema), defaultValues, mode: 'onChange' });
  const build = useMutation({
    mutationFn: (input: { file: File } | { values: ProfileValues }) => {
      if ('file' in input) return reportFromCv(input.file);
      const { skills, text } = profileRequest(input.values);
      return reportFromProfile(skills, text);
    },
    onSuccess: setReport,
  });
  useEffect(() => {
    if (Object.keys(form.formState.errors).length > 0) void form.trigger();
  }, [schema, form]);
  const values = form.watch();
  const allValid = schema.safeParse(values).success;
  const failure = build.error ? errorMessage(build.error, Boolean(cv)) : null;
  const requestId = build.error instanceof ApiError && build.error.status >= 500 ? build.error.requestId : null;

  if (report) {
    return <ReportView report={report} onEdit={() => { setReport(null); build.reset(); }} />;
  }

  const fields: { name: keyof ProfileValues; label: string; hint?: string; multi?: boolean; numeric?: boolean }[] = [
    { name: 'name', label: 'Name' },
    { name: 'skills', label: 'Description of skills', hint: 'What you can do today, in your own words.', multi: true },
    { name: 'certificates', label: 'Certificates', hint: 'Enter None if not applicable.' },
    { name: 'degree', label: 'University degree', hint: 'Enter None if not applicable.' },
    { name: 'age', label: 'Age', numeric: true },
  ];

  return (
    <Form {...form}>
      <form className="profile-form" noValidate onSubmit={(event) => {
        if (build.isPending) {
          event.preventDefault();
        } else if (cv) {
          event.preventDefault();
          build.mutate({ file: cv });
        } else {
          void form.handleSubmit((submitted) => build.mutate({ values: submitted }))(event);
        }
      }} data-testid="form-profile">
        <CvUpload file={cv} onChange={(file) => { setCv(file); if (file) form.clearErrors(); }} />
        <fieldset className="profile-manual" disabled={Boolean(cv)}>
          <legend>{t('Or enter your details manually')}</legend>
          <p className="profile-hint">{t(cv ? 'Remove the PDF to use the manual form instead.' : 'Complete all five fields if you are not using a CV PDF.')}</p>
          <div className="profile-fields">
        {fields.map((f, i) => (
          <FormField key={f.name} control={form.control} name={f.name} render={({ field }) => (
            <FormItem className={`profile-field ${f.multi ? 'wide' : ''}`}>
              <FormLabel className="profile-label"><span>{String(i + 1).padStart(2, '0')}</span>{t(f.label)}</FormLabel>
              <FormControl>
                {f.multi
                  ? <Textarea {...field} rows={4} className="profile-input" required={!cv} data-testid={`input-${f.name}`} />
                  : <Input {...field} className="profile-input" required={!cv} inputMode={f.numeric ? 'numeric' : undefined} autoComplete={f.name === 'name' ? 'name' : 'off'} data-testid={`input-${f.name}`} />}
              </FormControl>
              {f.hint && <FormDescription className="profile-hint">{t(f.hint)}</FormDescription>}
              {!cv && <FormMessage className="profile-error" data-testid={`error-${f.name}`} />}
            </FormItem>
          )} />
        ))}
          </div>
        </fieldset>
        <div className="profile-actions">
          <p className="profile-privacy" data-testid="text-privacy-note">{t(cv
            ? 'Your PDF is sent to our server, where its text is read to build your report.'
            : 'Your skills, certificates and degree are sent to our server to build your report. Your name and age stay on this page.')}</p>
          {(cv || allValid) && (
            <button type="submit" className="profile-submit" disabled={build.isPending} aria-busy={build.isPending} data-testid="button-submit-profile">
              {t(build.isPending ? 'Building your report…' : 'Submit profile')} <ArrowRight size={16} aria-hidden="true" />
            </button>
          )}
          {failure && (
            <p className="profile-error" role="alert" data-testid="text-report-error">
              {t(failure)}{requestId ? ` (${t('reference')} ${requestId})` : ''}
            </p>
          )}
        </div>
      </form>
    </Form>
  );
}
