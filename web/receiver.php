<?php
/**
 * Decisions receiver for the Town Square brick review pages.
 *
 * Accepts POSTed decisions.csv content from review.html (autosave and
 * final submit) and stores it OUTSIDE the web root, one timestamped file
 * per final submit and one rolling file per reviewer for autosaves.
 * Pull the files down with SFTP/rsync and feed them to apply_decisions.py.
 *
 * Deploy (see DEPLOY_DREAMHOST.md):
 *   1. Put this file in the subdomain's web root next to review.html.
 *   2. Create the data directory OUTSIDE the web root and set DATA_DIR.
 *   3. Change TOKEN to a long random string; give the same value to
 *      make_review_page.py --receiver-token.
 *
 * Requests:
 *   POST receiver.php?token=...&label=<reviewer>&final=0|1
 *     body: text/csv (the decisions.csv content)
 *   GET  receiver.php?token=...&action=list
 *     -> JSON [{"name":...,"size":...,"mtime":...}, ...]
 *   GET  receiver.php?token=...&action=fetch&name=<file>
 *     -> the stored CSV (name must be a bare .csv filename)
 * Responses: JSON {"ok":true,...} / {"ok":false,...} or the CSV body.
 * The GET actions let run_pipeline.py pull decisions without SFTP.
 */

// ---- configuration ---------------------------------------------------
const TOKEN    = 'CHANGE-ME-to-a-long-random-string';
const DATA_DIR = __DIR__ . '/../../brick_data';   // outside the web root
const MAX_BYTES = 2 * 1024 * 1024;                // decisions.csv is tiny
// ----------------------------------------------------------------------

header('Content-Type: application/json');

function fail(int $code, string $why): void {
    http_response_code($code);
    echo json_encode(['ok' => false, 'error' => $why]);
    exit;
}

if (!hash_equals(TOKEN, (string)($_GET['token'] ?? ''))) {
    fail(403, 'bad token');
}

if ($_SERVER['REQUEST_METHOD'] === 'GET') {
    $action = (string)($_GET['action'] ?? '');
    if ($action === 'list') {
        $out = [];
        foreach (glob(DATA_DIR . '/*.csv') ?: [] as $path) {
            $out[] = ['name' => basename($path),
                      'size' => filesize($path),
                      'mtime' => date('c', filemtime($path))];
        }
        echo json_encode($out);
        exit;
    }
    if ($action === 'fetch') {
        $name = (string)($_GET['name'] ?? '');
        // Bare .csv filenames only -- no separators, no traversal.
        if (!preg_match('/^[A-Za-z0-9._-]+\.csv$/', $name)
                || strpos($name, '..') !== false) {
            fail(400, 'bad name');
        }
        $path = DATA_DIR . '/' . $name;
        if (!is_file($path)) {
            fail(404, 'no such file');
        }
        header('Content-Type: text/csv');
        readfile($path);
        exit;
    }
    fail(400, 'unknown action');
}

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    fail(405, 'POST only');
}

$body = file_get_contents('php://input', false, null, 0, MAX_BYTES + 1);
if ($body === false || strlen($body) === 0) {
    fail(400, 'empty body');
}
if (strlen($body) > MAX_BYTES) {
    fail(413, 'too large');
}
// Must look like a decisions.csv: its header row names these columns.
$first = strtolower(strtok($body, "\r\n"));
if (strpos($first, 'reviewer') === false || strpos($first, 'decision') === false) {
    fail(400, 'not a decisions.csv');
}

if (!is_dir(DATA_DIR) && !mkdir(DATA_DIR, 0700, true)) {
    fail(500, 'data dir unavailable');
}

$label = preg_replace('/[^A-Za-z0-9_-]/', '', (string)($_GET['label'] ?? ''));
$label = $label !== '' ? substr($label, 0, 40) : 'anon';
$final = ($_GET['final'] ?? '0') === '1';

if ($final) {
    // Every final submit is kept forever, never overwritten.
    $name = sprintf('decisions_%s_%s_%s.csv',
                    $label, date('Ymd_His'), bin2hex(random_bytes(3)));
} else {
    // Autosaves roll over per reviewer -- the latest state is what counts,
    // and a click per brick would otherwise pile up thousands of files.
    $name = sprintf('autosave_%s.csv', $label);
}

$path = DATA_DIR . '/' . $name;
if (file_put_contents($path, $body, LOCK_EX) === false) {
    fail(500, 'write failed');
}

echo json_encode(['ok' => true, 'stored' => $name]);
