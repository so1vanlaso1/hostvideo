// Test the upload controls as code; no browser, network or GPU is used.
import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";

let extension, selectedFiles = [], pickerDone;
const alerts = [], uploads = [];
const imageNames = ["existing-outfit.png"];
const videoNames = [];
const schema = () => ({ character_image: [["Keep original", ...imageNames]],
    background_image: [["Keep original", ...imageNames]], reference_video: [["Choose dance video", ...videoNames]] });
class FormData {
    values = {};
    append(key, value) { this.values[key] = value; }
}
const sandbox = {
    app: { registerExtension(value) { extension = value; } },
    api: { async fetchApi(path, options) {
        if (path === "/upload/image") {
            const file = options.body.values.image;
            uploads.push(options.body.values);
            const name = `dance-assets/${file.name}`;
            (file.name.endsWith(".mp4") ? videoNames : imageNames).push(name);
            return { ok: true, json: async () => ({ name: file.name, subfolder: "dance-assets" }) };
        }
        assert.equal(path, "/object_info/H3DanceWorkflow");
        return { ok: true, json: async () => ({ H3DanceWorkflow: { input: { required: schema() } } }) };
    } },
    FormData,
    alert(message) { alerts.push(message); },
    document: { createElement(kind) {
        assert.equal(kind, "input");
        return { files: selectedFiles, click() { pickerDone = this.onchange(); } };
    } },
};
const source = fs.readFileSync(new URL("../custom_nodes/h3_lowvram/web/dance_workflow.js", import.meta.url), "utf8")
    .replace(/^import .*;\n/gm, "");
vm.runInNewContext(source, sandbox);
class Node {
    constructor() {
        const preset = JSON.parse(fs.readFileSync(new URL("../h3_pipeline/assets/workflows/dance_general_ui.json", import.meta.url))).nodes[0];
        const names = ["reference_video", "character_image", "background_image", "outfit_images", "prompt", "seed", "megapixels",
            "max_section_seconds", "audio_mode", "profile", "steps", "turbo", "context_frames", "outfit_durations", "start_seconds", "duration_seconds"];
        this.widgets = names.map((name, i) => ({ name, value: preset.widgets_values[i], options: {} }));
    }
    addWidget(type, name, value, callback, options) {
        const widget = { type, name, value, callback, options };
        this.widgets.push(widget);
        return widget;
    }
    setDirtyCanvas() {}
}
await extension.beforeRegisterNodeDef(Node, { name: "H3DanceWorkflow", input: { required: schema() } });
const node = new Node();
node.onNodeCreated();
const widget = name => node.widgets.find(w => w.name === name);
const click = async name => { await widget(name).callback(); if (pickerDone) { await pickerDone; pickerDone = null; } };

await click("Add selected existing outfit");
assert.deepEqual(JSON.parse(widget("outfit_images").value), ["existing-outfit.png"]);
selectedFiles = [{ name: "dress-1.webp" }, { name: "dress-2.png" }];
await click("Add outfit images (multiple)");
assert.deepEqual(JSON.parse(widget("outfit_images").value), ["existing-outfit.png", "dance-assets/dress-1.webp", "dance-assets/dress-2.png"]);
assert.equal(widget("character_image").value, "Keep original");
assert.equal(widget("background_image").value, "Keep original");
for (const [control, field, name] of [
    ["Upload dance video", "reference_video", "dance.mp4"],
    ["Upload replacement character", "character_image", "person.jpg"],
    ["Upload replacement background", "background_image", "scene.jpg"],
]) {
    selectedFiles = [{ name }];
    await click(control);
    assert.equal(widget(field).value, `dance-assets/${name}`);
    assert.ok(widget(field).options.values.includes(`dance-assets/${name}`));
}
assert.ok(uploads.every(body => body.type === "input" && body.overwrite === "false" && body.subfolder === "dance-assets"));
await click("Clear outfit list");
assert.equal(widget("outfit_images").value, "[]");
assert.equal(node.widgets.filter(w => w.serialize !== false).length, 16);
assert.deepEqual(alerts, []);
console.log("Dance upload controls passed: ordered multi-upload, existing outfits, independent replacements, video selection, clearing, serialization.");
