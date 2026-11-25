"""Draw the check sample for the caliper detector and write a local labelling page.

Usage: python -I make_marker_check.py <data/external> <manifest.csv> <out_dir> [<earlier_key.csv> [<dataset,dataset>]]

A simple random sample of N_PER_DATASET groups per dataset (one image per group) from all used rows.
Images are copied under neutral names in shuffled order, so the labeller sees neither the file name nor
the diagnosis. key.csv maps the neutral names back to sample_id; it holds no diagnosis label.
With an earlier key.csv, groups already drawn there are left out (fresh sample after a detector revision);
the optional dataset list restricts the draw. With MARKER_ALL=1 in the environment every image of the chosen
datasets is taken (full hand labelling) and only the images already in the earlier key are left out; the order
is random, so the first k images labelled are a random sample.
"""
import sys, os, csv, json, hashlib, shutil, collections
ALL = os.environ.get('MARKER_ALL') == '1'

R, MAN, OUT = sys.argv[1:4]
SEED = 20261123
N_PER_DATASET = 150


def h(*a):
    return hashlib.sha256('|'.join(map(str, a)).encode()).hexdigest()


rows = [r for r in csv.DictReader(open(MAN)) if r['experimental_split'] != 'excluded']
if len(sys.argv) > 4:
    seen_ids = {r['sample_id'] for r in csv.DictReader(open(sys.argv[4]))}
    seen = {(r['dataset'], r['group_id']) for r in rows if r['sample_id'] in seen_ids}
    rows = [r for r in rows if (r['sample_id'] not in seen_ids if ALL else (r['dataset'], r['group_id']) not in seen)]
if len(sys.argv) > 5:
    rows = [r for r in rows if r['dataset'] in sys.argv[5].split(',')]
by = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows: by[r['dataset']][r['group_id']].append(r)
picked = []
for ds in sorted(by):
    groups = sorted(by[ds], key=lambda g: h(SEED, 'group', g))[:N_PER_DATASET]
    if ALL:
        picked += [r for g in by[ds] for r in by[ds][g]]; continue
    for g in groups:
        picked.append(min(by[ds][g], key=lambda r: h(SEED, 'image', r['sample_id'])))
picked.sort(key=lambda r: h(SEED, 'order', r['sample_id']))

os.makedirs(os.path.join(OUT, 'img'), exist_ok=True)
names = []
with open(os.path.join(OUT, 'key.csv'), 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['item', 'file', 'sample_id', 'dataset'])
    for i, r in enumerate(picked, 1):
        ext = os.path.splitext(r['data_path'])[1].lower()
        name = f'{i:04d}{ext}'
        shutil.copyfile(os.path.join(R, r['data_path']), os.path.join(OUT, 'img', name))
        w.writerow([i, name, r['sample_id'], r['dataset']]); names.append(name)

PAGE = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Caliper check</title>
<style>
body{margin:0;background:#111;color:#ddd;font:15px/1.4 system-ui,sans-serif}
#top{display:flex;gap:16px;align-items:center;padding:8px 14px;background:#1c1c1c;flex-wrap:wrap}
#wrap{display:flex;justify-content:center;align-items:center;height:calc(100vh - 150px);padding:6px}
#im{max-width:100%;max-height:100%;min-height:70%;object-fit:contain;image-rendering:auto;background:#000}
button{font:inherit;padding:7px 14px;border-radius:6px;border:1px solid #555;background:#2a2a2a;color:#eee;cursor:pointer}
button.sel{outline:3px solid #4ea1ff}
#yes{border-color:#3c9a5f}#no{border-color:#b05a5a}
#help{padding:6px 14px;color:#aaa;font-size:13px}
#bar{height:5px;background:#333}#fill{height:5px;background:#4ea1ff;width:0}
kbd{background:#333;border-radius:4px;padding:1px 6px}
</style></head><body>
<div id="top"><b id="pos"></b>
<button id="yes">Calipers present <kbd>Y</kbd></button>
<button id="no">No calipers <kbd>N</kbd></button>
<button id="unsure">Cannot tell <kbd>U</kbd></button>
<button id="back">Back <kbd>&larr;</kbd></button><button id="fwd">Next <kbd>&rarr;</kbd></button>
<button id="first">First unlabelled</button>
<button id="save">Download labels (CSV)</button><span id="done"></span></div>
<div id="bar"><div id="fill"></div></div>
<div id="help"><b>Calipers</b> = measurement marks drawn on the picture: small <b>+</b> or <b>x</b> crosses, usually in pairs,
sometimes joined by a dotted line or numbered. Answer yes if at least one is visible anywhere.
<b>Do not count</b> the depth scale at the edge, body or probe icons, text, the colour Doppler box, or bright spots of the tissue itself.
A dotted box or dotted line <b>without</b> any + or x is <b>no</b>. Coloured blood-flow patches are <b>no</b>. A coloured + or x is <b>yes</b>.
Answers are kept in this browser; you can stop at any time and come back. Download the CSV when finished (or at the end of each sitting, as a backup) and save it next to this page as <code>marker_labels.csv</code>.</div>
<div id="wrap"><img id="im" alt=""></div>
<script>
const FILES=__FILES__, KEY='caliper_check___TAG__';
let L={}; try{L=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){}
let i=0; const $=id=>document.getElementById(id);
function firstOpen(){const k=FILES.findIndex(f=>!L[f]);return k<0?FILES.length-1:k}
function show(){const f=FILES[i];$('im').src='img/'+f;$('pos').textContent=(i+1)+' / '+FILES.length;
 for(const b of ['yes','no','unsure'])$(b).classList.toggle('sel',L[f]===b);
 const n=Object.keys(L).length;$('fill').style.width=(100*n/FILES.length)+'%';
 $('done').textContent=n+' labelled'+(n===FILES.length?' — all done, download the CSV':'');}
function set(v){L[FILES[i]]=v;try{localStorage.setItem(KEY,JSON.stringify(L))}catch(e){}
 if(i<FILES.length-1)i++;show()}
function save(){let s='file,calipers\n';for(const f of FILES)s+=f+','+(L[f]||'')+'\n';
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([s],{type:'text/csv'}));
 a.download='marker_labels.csv';a.click()}
$('yes').onclick=()=>set('yes');$('no').onclick=()=>set('no');$('unsure').onclick=()=>set('unsure');
$('back').onclick=()=>{if(i>0)i--;show()};$('fwd').onclick=()=>{if(i<FILES.length-1)i++;show()};
$('first').onclick=()=>{i=firstOpen();show()};$('save').onclick=save;
document.addEventListener('keydown',e=>{const k=e.key.toLowerCase();
 if(k==='y')set('yes');else if(k==='n')set('no');else if(k==='u')set('unsure');
 else if(e.key==='ArrowLeft')$('back').click();else if(e.key==='ArrowRight')$('fwd').click()});
i=firstOpen();show();
</script></body></html>'''
tag = h(SEED, *names)[:10]
open(os.path.join(OUT, 'label.html'), 'w').write(PAGE.replace('__FILES__', json.dumps(names)).replace('__TAG__', tag))
print('images', len(names), dict(collections.Counter(r['dataset'] for r in picked)),
      '| partitions', dict(collections.Counter(r['experimental_split'] for r in picked)))
print('key sha256', hashlib.sha256(open(os.path.join(OUT, 'key.csv'), 'rb').read()).hexdigest())
