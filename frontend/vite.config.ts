import path from 'node:path';
import { fileURLToPath } from 'node:url';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import { defineConfig, type Plugin } from 'vite';

const root = path.dirname(fileURLToPath(import.meta.url));
const port = Number(process.env.PORT ?? 5173);
if (!Number.isInteger(port) || port < 1 || port > 65535) {
  throw new Error('PORT must be an integer between 1 and 65535.');
}
const host = process.env.HOST ?? 'localhost';
const allowedHosts = process.env.ALLOWED_HOSTS?.split(',').map((h) => h.trim()).filter(Boolean);
const apiOrigin = process.env.VITE_API_URL ? new URL(process.env.VITE_API_URL).origin : '';
const apiProxy = { '/api': process.env.API_PROXY ?? 'http://localhost:8000' };

const contentSecurityPolicy = [
  "default-src 'self'",
  "script-src 'self'",
  "style-src 'self' https://fonts.googleapis.com",
  "font-src 'self' https://fonts.gstatic.com",
  "img-src 'self' data:",
  `connect-src 'self'${apiOrigin ? ` ${apiOrigin}` : ''}`,
  "object-src 'none'",
  "base-uri 'self'",
  "form-action 'self'",
  'upgrade-insecure-requests',
].join('; ');

function securityMeta(): Plugin {
  return {
    name: 'security-meta',
    apply: 'build',
    transformIndexHtml: () => [
      { tag: 'meta', attrs: { 'http-equiv': 'Content-Security-Policy', content: contentSecurityPolicy }, injectTo: 'head-prepend' },
    ],
  };
}

export default defineConfig({
  root,
  base: process.env.BASE_PATH ?? '/',
  plugins: [react(), tailwindcss(), securityMeta()],
  resolve: {
    alias: { '@': path.resolve(root, 'src') },
    dedupe: ['react', 'react-dom'],
  },
  build: {
    outDir: path.resolve(root, 'dist'),
    emptyOutDir: true,
  },
  server: {
    host,
    port,
    strictPort: true,
    allowedHosts,
    proxy: apiProxy,
  },
  preview: {
    host,
    port,
    strictPort: true,
    allowedHosts,
    proxy: apiProxy,
  },
});
