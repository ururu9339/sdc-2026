# currvix.ai frontend

The Currvix website. It talks to the FastAPI backend in this repository through
`/api/v1`. In production FastAPI serves the built `dist/` folder, so the site and
the API share one origin.

## Requirements

- Node.js 20.19+ (or Node.js 22.12+).
- pnpm 10.

## Run

```sh
cd frontend
pnpm install
pnpm dev
```

The development server uses port 5173 by default and forwards `/api` to the
backend at `http://localhost:8000`, so start the API first:

```sh
uvicorn src.backend.main:app --port 8000   # from the repository root
```

## Check and build

```sh
pnpm typecheck
pnpm build
pnpm preview
```

The production build is written to `dist/`. Configure your static host to serve
`index.html` for application routes such as `/explore`.

Optional environment variables:

- `PORT`: server port (default `5173`).
- `BASE_PATH`: URL prefix if hosted in a subdirectory (default `/`).
- `HOST`: interface the dev and preview servers listen on (default `localhost`). Set `0.0.0.0` only when
  the server must be reachable from other machines, such as on Replit.
- `ALLOWED_HOSTS`: comma-separated host names the dev and preview servers accept when `HOST` is not
  `localhost`, for example `.replit.dev`. Unset keeps Vite's DNS-rebinding protection.
- `API_PROXY`: backend the dev and preview servers forward `/api` to (default
  `http://localhost:8000`).
- `VITE_API_URL`: build-time API origin, only when the API is hosted on another
  origin. It is added to the Content-Security-Policy `connect-src`.

## Security

The production build adds a Content-Security-Policy meta tag (scripts and data from this origin only,
styles and fonts also from Google Fonts; `VITE_API_URL`, when set, is added to `connect-src`). FastAPI
sends these headers with every response, including the site; any other static host must send them too:

```
Content-Security-Policy: frame-ancestors 'none'
X-Content-Type-Options: nosniff
Referrer-Policy: strict-origin-when-cross-origin
Strict-Transport-Security: max-age=31536000
```

IBM Plex Mono is loaded from Google Fonts, so the font requires an internet connection.

## Contents

- `src/`: both pages, animations, bilingual text, form validation, CV PDF selection,
  reusable components, and styles.
- `public/`: favicon and robots file.
- `index.html`: app entry page.
- `components.json`: component-library configuration.
- `package.json` and `pnpm-lock.yaml`: standalone dependencies and install lockfile.
- `vite.config.ts` and `tsconfig.json`: local build and TypeScript configuration.

## Current behavior

English and Bahasa Melayu are supported. Submitting the form sends the skills,
certificates and degree to `POST /api/v1/report`; the name and age stay in the
browser. A CV PDF is checked locally, then sent to `POST /api/v1/candidates/pdf`
for OCR, and the report comes from `GET /api/v1/report/candidate/{id}`. The
Explore page also shows emerging and declining skills and jobs from
`GET /api/v1/forecast`. Reports follow the rendering rules in
`docs/API_FOR_FRONTEND.md`: the preview banner, outlook chips with their dates,
and the out-of-date marker are always shown.
