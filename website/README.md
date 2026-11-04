# Forecast website

A phone-friendly, password-protected page comparing the latest forecasts, rebuilt every 3 hours
by GitHub Actions ([`.github/workflows/forecast-site.yml`](../.github/workflows/forecast-site.yml)) and
hosted on GitHub Pages.

- [`index.html`](index.html) is the page. It holds no forecast data, so it's fine to be public.
- [`build.py`](build.py) refreshes the forecast store and writes `forecast.bin`: the data the page
  shows, encrypted with the site password (AES-256-GCM, key from PBKDF2-SHA256). The page decrypts it
  in the browser. Anyone can download `forecast.bin` and guess passwords offline, so use a long
  passphrase (at least 12 characters, ideally four or more random words).
- Maps cover the UK (every source averaged onto a 0.5° grid, so they can be compared and averaged) or
  London (each source at its own resolution, sampled onto a ~2 km grid, with borough outlines and the
  Thames; colours scaled to London's range). Charts are for five areas of London and five UK cities.
  Map outlines come from Natural Earth, written by [`outlines.py`](outlines.py).
- Source picking, the means (at each time, the average of the picked sources that forecast it) and
  the maps are all worked out in the browser, so they update as you pick sources.

## Setting up

1. Create an **empty** repository on GitHub. GitHub Pages is only free for **public** repositories;
   that's fine here, as forecast data is never committed: `data/` is ignored, and notebook outputs are
   stripped when committing.
2. In the repository's **Settings → Secrets and variables → Actions**, add two secrets:
   - `MET_OFFICE_API_KEY`: the same key as in `.env`
   - `SITE_PASSWORD`: the site's password
3. In **Settings → Pages**, set **Source** to **GitHub Actions**.
4. Commit and push (from the project root):
   ```sh
   git add -A
   git commit -m "Forecast website"
   git remote add origin https://github.com/<you>/<repo>.git
   git push -u origin main
   ```
5. In the **Actions** tab, run **Forecast website** (**Run workflow**) rather than waiting for the
   schedule. The first build downloads everything, so takes a while; later ones only download new
   runs. The site is then at `https://<you>.github.io/<repo>/`.

On your phone, tick **Remember on this device** when unlocking, and use **Add to Home Screen** for an
app-like icon. **Lock** forgets the password on that device.

## Notebook outputs

Notebook outputs contain forecast data (including the Met Office's), so `.gitattributes` strips them
on commit. This uses the `strip-output` git filter, which is set up in this clone's `.git/config`. In a
new clone, set it up again before committing any notebook:

```sh
git config filter.strip-output.clean ".venv/bin/jupyter nbconvert --ClearOutputPreprocessor.enabled=True --to notebook --stdin --stdout --log-level=ERROR"
git config filter.strip-output.smudge cat
git config filter.strip-output.required true
```

## Building locally

With `SITE_PASSWORD` in `.env`:

```sh
python -m website.build --no-refresh --out _site   # from the stored forecasts, downloading nothing
cd _site && python -m http.server 8000             # then open http://localhost:8000
```

The page needs `localhost` or HTTPS (browsers only allow the decryption there). Changing the
password takes effect at the next build; phones that remembered the old one ask again.
