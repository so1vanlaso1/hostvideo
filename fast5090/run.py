#!/usr/bin/env python3
"""Lean RTX5090 ComfyUI client. No dependency on the legacy h3_pipeline."""
import argparse, json, pathlib, time, urllib.request, urllib.error, uuid

HERE = pathlib.Path(__file__).resolve().parent
def api(method, path, body=None):
    raw = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request("http://127.0.0.1:8188"+path, data=raw, method=method,
        headers={"Content-Type":"application/json"} if raw else {})
    with urllib.request.urlopen(req, timeout=45) as resp:
        return json.load(resp)
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--preset",choices=["pdd8","vsa4"],default="pdd8")
    p.add_argument("--prompt",default=None)
    p.add_argument("--image",default="reference.png",help="Image already uploaded into ComfyUI input directory")
    p.add_argument("--width",type=int,default=1344)
    p.add_argument("--height",type=int,default=768)
    p.add_argument("--frames",type=int,default=124)
    p.add_argument("--seed",type=int,default=123456789)
    p.add_argument("--timeout",type=int,default=3600)
    args=p.parse_args()
    if args.width%32 or args.height%32 or args.frames not in (73,124,243,362):
        p.error("width/height must be divisible by 32; frames must be 73, 124, 243 or 362")
    graph=json.loads((HERE/"workflows"/(args.preset+"_api.json")).read_text())
    if args.preset=="pdd8":
        condition,noise,image="7","10","19"
    else:
        condition,noise,image="6","8","50"
    graph[condition]["inputs"].update(width=args.width,height=args.height,length=args.frames)
    if args.prompt:
        graph[condition]["inputs"]["prompt"]=args.prompt
    graph[noise]["inputs"]["noise_seed"]=args.seed
    graph[image]["inputs"]["image"]=args.image
    # Fail early on missing custom nodes/checkpoints, before expensive generation.
    info=api("GET","/object_info")
    missing=sorted({n["class_type"] for n in graph.values()}-set(info))
    if missing:
        raise SystemExit(f"ComfyUI is missing node types: {missing}. Check setup and restart.")
    started=time.monotonic()
    response=api("POST","/prompt",{"prompt":graph,"client_id":str(uuid.uuid4())})
    if response.get("node_errors"):
        raise SystemExit("ComfyUI node validation errors: "+json.dumps(response["node_errors"],indent=2))
    prompt_id=response["prompt_id"]
    print(f"Submitted {args.preset}: {prompt_id}",flush=True)
    while time.monotonic()-started < args.timeout:
        history=api("GET","/history/"+prompt_id).get(prompt_id)
        if history:
            status=history.get("status",{})
            if status.get("status_str")=="error" or status.get("completed") is False:
                raise SystemExit("Generation failed: "+json.dumps(status,indent=2))
            print(json.dumps({"prompt_id":prompt_id,"elapsed_seconds":round(time.monotonic()-started,1),
                             "outputs":history.get("outputs",{}),"status":status},indent=2))
            return
        time.sleep(3)
    raise SystemExit(f"Timeout; job {prompt_id} may still be running. Inspect /history/{prompt_id} before resubmitting.")
if __name__=="__main__":
    main()
