import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const KEEP = "Keep original";

function outfitList(widget) {
    const text = (widget.value || "").trim();
    const names = text.startsWith("[") ? JSON.parse(text) : text.split(/\r?\n/).map(s => s.trim()).filter(Boolean);
    if (!Array.isArray(names) || names.some(name => typeof name !== "string")) {
        throw new Error("Use a JSON array or one uploaded outfit filename per line.");
    }
    return names;
}

async function upload(file) {
    const body = new FormData();
    body.append("image", file);
    body.append("type", "input");
    body.append("subfolder", "dance-assets");
    body.append("overwrite", "false");
    const response = await api.fetchApi("/upload/image", { method: "POST", body });
    if (!response.ok) throw new Error(`Upload failed (${response.status}): ${await response.text()}`);
    const saved = await response.json();
    return saved.subfolder ? `${saved.subfolder}/${saved.name}` : saved.name;
}

app.registerExtension({
    name: "hostvideo.generalDance",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "H3DanceWorkflow") return;
        const original = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            original?.apply(this, arguments);
            const node = this;
            const widget = name => node.widgets.find(w => w.name === name);
            let imageFiles = nodeData.input.required.character_image[0].filter(name => name !== KEEP);
            const existing = node.addWidget("combo", "Existing outfit", imageFiles[0] || "No uploaded images",
                () => {}, { values: imageFiles.length ? imageFiles : ["No uploaded images"], serialize: false });
            existing.serialize = false;
            const button = (name, callback) => {
                const control = node.addWidget("button", name, null, async () => {
                    try { await callback(); }
                    catch (error) { alert(error.message); }
                    node.setDirtyCanvas(true, true);
                }, { serialize: false });
                control.serialize = false;
            };
            const refresh = async () => {
                const response = await api.fetchApi("/object_info/H3DanceWorkflow");
                if (!response.ok) throw new Error("Could not refresh the uploaded file list.");
                const schema = (await response.json()).H3DanceWorkflow.input.required;
                for (const name of ["reference_video", "character_image", "background_image"]) {
                    widget(name).options.values = schema[name][0];
                }
                imageFiles = schema.character_image[0].filter(name => name !== KEEP);
                existing.options.values = imageFiles.length ? imageFiles : ["No uploaded images"];
                if (!existing.options.values.includes(existing.value)) existing.value = existing.options.values[0];
            };
            const chooseFiles = (accept, multiple, callback) => {
                const picker = document.createElement("input");
                picker.type = "file";
                picker.accept = accept;
                picker.multiple = multiple;
                picker.onchange = async () => {
                    try {
                        const saved = [];
                        for (const file of picker.files) saved.push(await upload(file));
                        await refresh();
                        callback(saved);
                        node.setDirtyCanvas(true, true);
                    } catch (error) { alert(error.message); }
                };
                picker.click();
            };
            button("Upload dance video", () => chooseFiles("video/*", false, names => widget("reference_video").value = names[0]));
            button("Upload replacement character", () => chooseFiles("image/*", false, names => widget("character_image").value = names[0]));
            button("Upload replacement background", () => chooseFiles("image/*", false, names => widget("background_image").value = names[0]));
            const addOutfits = names => widget("outfit_images").value = JSON.stringify([...outfitList(widget("outfit_images")), ...names], null, 2);
            button("Add outfit images (multiple)", () => chooseFiles("image/*", true, addOutfits));
            button("Add selected existing outfit", () => {
                if (!imageFiles.includes(existing.value)) throw new Error("Upload an outfit image first.");
                addOutfits([existing.value]);
            });
            button("Clear outfit list", () => widget("outfit_images").value = "[]");
            if (widget("anchor_manifest")) {
                const anchorTime = node.addWidget("number", "Edited anchor time (seconds)", 0, () => {},
                    { min: 0, step: 0.1, precision: 3, serialize: false });
                const anchorOutfit = node.addWidget("number", "Edited anchor outfit number", 1, () => {},
                    { min: 1, step: 1, precision: 0, serialize: false });
                anchorTime.serialize = anchorOutfit.serialize = false;
                button("Add edited pose/outfit anchor", () => chooseFiles("image/*", false, names => {
                    const frame = Math.round(anchorTime.value * 24);
                    const outfit_index = Math.round(anchorOutfit.value);
                    let anchors = JSON.parse(widget("anchor_manifest").value || "[]");
                    if (!Array.isArray(anchors)) throw new Error("anchor_manifest must be a JSON array.");
                    anchors = anchors.filter(a => a.frame !== frame || a.outfit_index !== outfit_index);
                    anchors.push({ frame, outfit_index, image: names[0] });
                    anchors.sort((a, b) => a.frame - b.frame || a.outfit_index - b.outfit_index);
                    widget("anchor_manifest").value = JSON.stringify(anchors, null, 2);
                }));
            }
            button("Refresh uploaded files", refresh);
        };
    },
});
