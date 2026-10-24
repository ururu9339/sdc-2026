import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from 'react';

type Language = 'en' | 'ms';
const malay: Record<string, string> = {
  'currvix.ai home': 'Laman utama currvix.ai',
  'Main navigation': 'Navigasi utama',
  'Language': 'Bahasa',
  'Explore roles': 'Terokai kerjaya',
  'Work is changing. So can you.': 'Dunia kerja berubah. Anda juga boleh berubah.',
  'Career choices shouldn’t feel like guesswork. Currvix makes the signals in a changing job market easier to see — and the next skill easier to name.': 'Pilihan kerjaya tidak sepatutnya bergantung pada tekaan. Currvix membantu anda memahami perubahan pasaran kerja — dan mengenal pasti kemahiran seterusnya.',
  'Explore the signal map': 'Terokai peta kerjaya',
  'How to get started': 'Cara untuk bermula',
  'Submission note': 'Nota penghantaran',
  'Click on Explore the signal map': 'Klik Terokai peta kerjaya',
  'Fill up the form or choose a CV PDF': 'Isi borang atau pilih CV dalam format PDF',
  'Click Submit profile': 'Klik Hantar profil',
  'The Submit profile button appears after you select a valid PDF or complete all five fields.': 'Butang Hantar profil muncul selepas anda memilih PDF yang sah atau melengkapkan kelima-lima ruangan.',
  'Upload your CV': 'Muat naik CV anda',
  'Choose a CV PDF instead of filling in the form. PDF only, up to 10 MB.': 'Pilih CV dalam format PDF sebagai ganti mengisi borang. PDF sahaja, sehingga 10 MB.',
  'Choose PDF': 'Pilih PDF',
  'Replace PDF': 'Ganti PDF',
  'Remove PDF': 'Buang PDF',
  'Checking your PDF…': 'Menyemak PDF anda…',
  'PDF selected. The manual form is not required.': 'PDF dipilih. Borang manual tidak diperlukan.',
  'Or enter your details manually': 'Atau isi butiran anda secara manual',
  'Remove the PDF to use the manual form instead.': 'Buang PDF untuk menggunakan borang manual.',
  'Complete all five fields if you are not using a CV PDF.': 'Lengkapkan kelima-lima ruangan jika anda tidak menggunakan CV PDF.',
  'Please choose a PDF file.': 'Sila pilih fail PDF.',
  'The PDF is empty. Please choose another file.': 'PDF ini kosong. Sila pilih fail lain.',
  'Please choose a PDF smaller than 10 MB.': 'Sila pilih PDF yang bersaiz kurang daripada 10 MB.',
  'This file does not appear to be a PDF.': 'Fail ini bukan fail PDF yang sah.',
  'The file could not be read. Please choose it again.': 'Fail tidak dapat dibaca. Sila pilih semula.',
  'For workers · students · career counsellors': 'Untuk pekerja · pelajar · kaunselor kerjaya',
  'Currvix / Signal map': 'Currvix / Peta kerjaya',
  'Find the work taking shape.': 'Temui kerjaya masa hadapan.',
  'Please enter your name.': 'Sila masukkan nama anda.',
  'Please describe your skills.': 'Sila terangkan kemahiran anda.',
  'Please list your certificates, or enter None.': 'Sila senaraikan sijil anda, atau masukkan Tiada.',
  'Please enter your university degree, or enter None.': 'Sila masukkan ijazah universiti anda, atau masukkan Tiada.',
  'Please enter your age.': 'Sila masukkan umur anda.',
  'Age must be a whole number.': 'Umur mestilah nombor bulat.',
  'Age must be between 1 and 120.': 'Umur mestilah antara 1 dan 120.',
  'Name': 'Nama',
  'Description of skills': 'Penerangan kemahiran',
  'Certificates': 'Sijil',
  'University degree': 'Ijazah universiti',
  'Age': 'Umur',
  'What you can do today, in your own words.': 'Apa yang anda boleh lakukan hari ini, dengan kata-kata anda sendiri.',
  'Enter None if not applicable.': 'Masukkan None atau Tiada jika tidak berkenaan.',
  'Submit profile': 'Hantar profil',
  'Edit details': 'Sunting butiran',
  'Prototype · unvalidated forecasts': 'Prototaip · ramalan belum disahkan',
  'Your PDF is sent to our server, where its text is read to build your report.': 'PDF anda dihantar ke pelayan kami, dan teksnya dibaca untuk menyediakan laporan anda.',
  'Your skills, certificates and degree are sent to our server to build your report. Your name and age stay on this page.': 'Kemahiran, sijil dan ijazah anda dihantar ke pelayan kami untuk menyediakan laporan. Nama dan umur anda kekal di halaman ini.',
  'Building your report…': 'Menyediakan laporan anda…',
  'reference': 'rujukan',
  'Something went wrong. Please try again.': 'Berlaku ralat. Sila cuba lagi.',
  'The service could not be reached. Please check your connection and try again.': 'Perkhidmatan tidak dapat dihubungi. Sila semak sambungan anda dan cuba lagi.',
  'We could not read this PDF right now. Please type your skills in the form instead.': 'Kami tidak dapat membaca PDF ini sekarang. Sila taip kemahiran anda dalam borang.',
  'Please shorten your skills list and try again.': 'Sila pendekkan senarai kemahiran anda dan cuba lagi.',
  'Your report': 'Laporan anda',
  'Matched with Qwen3 embeddings': 'Dipadankan dengan embedding Qwen3',
  'Matched with the offline skill dictionary': 'Dipadankan dengan kamus kemahiran luar talian',
  'Skills we recognised': 'Kemahiran yang kami kenal pasti',
  'None yet. Try listing your skills separated by commas.': 'Belum ada. Cuba senaraikan kemahiran anda dipisahkan dengan koma.',
  'Not recognised': 'Tidak dikenal pasti',
  'Jobs closest to your profile': 'Pekerjaan paling hampir dengan profil anda',
  'overlap': 'padanan',
  'Missing': 'Belum dimiliki',
  'Your reskilling path': 'Laluan peningkatan kemahiran anda',
  'Start here': 'Mula di sini',
  'Next': 'Seterusnya',
  'Later': 'Kemudian',
  'Past share of postings': 'Bahagian iklan jawatan sebelum ini',
  'Descriptive history from our data, not a forecast.': 'Sejarah deskriptif daripada data kami, bukan ramalan.',
  'Rising': 'Meningkat',
  'Falling': 'Menurun',
  'Stable': 'Stabil',
  'Rising (unvalidated)': 'Meningkat (belum disahkan)',
  'Falling (unvalidated)': 'Menurun (belum disahkan)',
  'Stable (unvalidated)': 'Stabil (belum disahkan)',
  'No outlook': 'Tiada unjuran',
  'unconfirmed': 'belum dipastikan',
  'out of date': 'data lama',
  'same signal as': 'isyarat sama dengan',
  'Same signal as': 'Isyarat sama dengan',
  'Market signals': 'Isyarat pasaran',
  'Skills and jobs whose share of postings is moving': 'Kemahiran dan pekerjaan yang bahagian iklannya sedang berubah',
  'Emerging and declining': 'Meningkat dan menurun',
  'Emerging': 'Meningkat',
  'Declining': 'Menurun',
  'Market signals are unavailable right now. Please try again later.': 'Isyarat pasaran tidak tersedia sekarang. Sila cuba lagi kemudian.',
  'Loading market signals…': 'Memuatkan isyarat pasaran…',
  'No matches for these filters.': 'Tiada padanan.',
  'Job': 'Pekerjaan',
  'Skill': 'Kemahiran',
  'Range': 'Julat',
  'to': 'hingga',
};

type LanguageContextValue = {
  language: Language;
  setLanguage: (language: Language) => void;
  t: (text: string) => string;
};
const LanguageContext = createContext<LanguageContextValue | null>(null);

export function LanguageProvider({ children }: { children: ReactNode }) {
  const [language, setLanguage] = useState<Language>(() => {
    try { return localStorage.getItem('currvix-language') === 'ms' ? 'ms' : 'en'; }
    catch { return 'en'; }
  });
  useEffect(() => {
    document.documentElement.lang = language;
    document.title = language === 'ms' ? 'Currvix — Kerjaya dan kemahiran masa hadapan' : 'Currvix — Tomorrow’s work, your next skill';
    try { localStorage.setItem('currvix-language', language); } catch { /* Browsing without storage still supports switching. */ }
  }, [language]);
  const t = useCallback((text: string) => language === 'ms' ? malay[text] ?? text : text, [language]);
  return <LanguageContext.Provider value={{ language, setLanguage, t }}>{children}</LanguageContext.Provider>;
}

export function useLanguage() {
  const context = useContext(LanguageContext);
  if (!context) throw new Error('LanguageProvider is required.');
  return context;
}
