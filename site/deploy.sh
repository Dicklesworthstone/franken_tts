#!/usr/bin/env bash
# Deploys the site with @SITEV@ stamped to the current commit (plus a timestamp), so
# every script URL in the chain (index.html -> app.js/viz.js -> loader/manifest/worker
# -> pkg) cache-busts on each deploy. This defends against the frankentts.com zone's
# 4-hour browser-cache TTL override, which ignores the origin's no-cache headers and
# once paired a fresh index.html with four-hour-old scripts (empty visualization boxes).
#
# Run from site/: ./deploy.sh
set -euo pipefail

cd "$(dirname "$0")"
VERSION="$(git rev-parse --short HEAD)-$(date +%s)"
STAGE="$(mktemp -d /tmp/frankentts-site-deploy.XXXXXX)"

rsync -a --exclude deploy.sh ./ "$STAGE/"
# Every file that CONTAINS the token must be listed, not just the entry points: kernel-worker.js
# imports ./pkg/ftts_wasm.js?v=@SITEV@ and would otherwise ship the literal placeholder, quietly
# opting the partition workers out of cache-busting while everything else rotated.
for f in "$STAGE"/index.html "$STAGE"/app.js "$STAGE"/loader.js "$STAGE"/engine-worker.js "$STAGE"/kernel-worker.js; do
  perl -pi -e "s/\@SITEV\@/$VERSION/g" "$f"
done

echo "deploying version $VERSION from $STAGE"
(cd "$STAGE" && wrangler pages deploy . --project-name frankentts --branch main --commit-dirty=true)
echo "staged copy left at $STAGE (temp dir; not auto-deleted)"
