# Hosting the brick pages on Dreamhost

Everything static plus one tiny PHP receiver. Uses the existing shared
hosting (PHP already enabled for WordPress) — no added cost.

## One-time setup

1. **Subdomain.** Dreamhost panel → Manage Websites → Add Website →
   `bricks.<your-domain>` (a subdomain avoids the WordPress rewrite rules).
   Enable **Let's Encrypt HTTPS** (free) on it.

2. **Directories.** Over SSH/SFTP, alongside the subdomain's web root:

   ```
   ~/bricks.<domain>/          <- web root (Dreamhost creates it)
       search.html
       review.html
       fp_review.html
       receiver.php
       .htaccess
   ~/brick_data/               <- OUTSIDE the web root; receiver writes here
   ```

   `receiver.php`'s `DATA_DIR` default (`../../brick_data`) matches this
   layout; adjust if yours differs.

3. **Basic auth** — create the password file and `.htaccess`:

   ```
   htpasswd -c ~/.brick_htpasswd parksstaff        # prompts for a password
   ```

   `~/bricks.<domain>/.htaccess`:

   ```
   AuthType Basic
   AuthName "Town Square bricks"
   AuthUserFile /home/<shell-user>/.brick_htpasswd
   Require valid-user
   ```

   One shared staff login is fine — the receiver token and reviewer names
   do the fine-grained accounting.

   **Pausing the password:** keep two local variants (git-ignored,
   `web/htaccess_auth_on.local` / `web/htaccess_auth_off.local`) and
   upload whichever one as `.htaccess` — that's the whole toggle. The
   "off" variant opens the search page and photos but keeps
   `review.html`, `fp_review.html` and `receiver.php` behind the login:
   generated review pages embed the receiver token in their HTML, so a
   public review page would leak the token to anyone who views source.

4. **Token.** Edit `receiver.php`, set `TOKEN` to a long random string
   (e.g. from `python -c "import secrets; print(secrets.token_urlsafe(32))"`).
   Use the same value when building review pages.

## Per-batch workflow

```powershell
# build the review page wired to the server
python make_review_page.py --review output/review_matched.csv `
    --photos photos/ --catalog output/singles.csv `
    --output output/review.html `
    --receiver-url https://bricks.<domain>/receiver.php `
    --receiver-token <the token>

# build / refresh the counter search page
python make_search_page.py --master reference/master_list.csv `
    --matched output/singles_matched.csv --output output/search.html
```

Upload both via SFTP. **Keep the stable filenames** (`search.html`,
`review.html`, `fp_review.html`): the pages' shared nav bar links them to
each other by these names, and one basic-auth login covers all three.
Stale-cache worries are handled by `.htaccess`, which serves every
`.html` with `Cache-Control: no-cache` (browser caching served reviewers
stale pages before that header existed — the old fix was date-stamped
filenames; the header replaced it). Each page shows its build date in
the nav bar, so staff can confirm they're seeing the latest upload.

Reviewers just get the URL (plus the shared login). Their clicks autosave
to the server every few seconds ("saved to server HH:MM" appears in the
header); the download-and-email button still works as a fallback.

## Collecting decisions

```powershell
# pull everything the receiver has stored
sftp <shell-user>@<server>:brick_data/*.csv output/decisions/
# or: rsync -av <shell-user>@<server>:brick_data/ output/decisions/

python apply_decisions.py --matched output/matched.csv `
    --decisions output/decisions/*.csv --output output/matched_final.csv
```

`decisions_<name>_<timestamp>_<rand>.csv` files are final submits (kept
forever, never overwritten); `autosave_<name>.csv` is each reviewer's
rolling latest state. Feed apply_decisions.py the finals, or the autosave
if a reviewer never pressed export — later files win on conflict.

## Photos: hosted review at scale

Embedded-photo review pages cap out at a few hundred items. For big
queues, upload derivative trees and build the page against them:

```powershell
# one-time per photo batch (resumable -- re-run after adding pallets)
python make_derivatives.py --input photos/ --output derivatives/
# one-time ever: the scanned brick list's rows as images (~350 MB)
python make_strips.py --pdf "TSP Bricks ALL - OG List by Name - OCR.pdf" `
    --output derivatives/strips
rsync -av derivatives/ <shell-user>@<server>:bricks.<domain>/photos/

# review page that references the hosted images (tiny at any size);
# --master adds each candidate's scanned-row image next to its text
python make_review_page.py --review output/review_matched.csv `
    --catalog output/singles.csv --master reference/master_list.csv `
    --photo-base-url https://bricks.<domain>/photos `
    --receiver-url https://bricks.<domain>/receiver.php `
    --receiver-token <the token> `
    --output output/review.html
```

Derivatives are ~2500px "zoom" JPEGs plus ~640px thumbs (~8-15 GB total),
NOT camera originals (40-100 GB; HEIC won't display in browsers anyway).
Reviewers see lazy-loaded thumbnails; clicking one opens the zoom image —
which is what worn bricks need. Dreamhost is a working copy, not the
archive — originals stay on the laptop/external drive.

Security note: the repo is public. The real receiver token lives ONLY in
the uploaded receiver.php and in generated review pages on the protected
subdomain — never in git (output/ and web/receiver.local.php are
git-ignored).
