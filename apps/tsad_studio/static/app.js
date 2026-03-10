const state = {
  defaults: null,
  config: null,
  ui: null,
  preview: null,
  selectedNodes: [],
  activeTab: "series",
  configVersion: 0,
  requestIds: {
    randomize: 0,
    preview: 0,
  },
  pending: {
    randomize: false,
    preview: false,
  },
};

const dom = {
  form: document.getElementById("config-form"),
  status: document.getElementById("status-banner"),
  restore: document.getElementById("restore-defaults"),
  randomize: document.getElementById("randomize-config"),
  preview: document.getElementById("preview-sample"),
  summaryChips: document.getElementById("summary-chips"),
  datasetSelect: document.getElementById("dataset-select"),
  maxNodesSelect: document.getElementById("max-nodes-select"),
  nodeSelector: document.getElementById("node-selector"),
  seriesCanvas: document.getElementById("series-canvas"),
  maskCanvas: document.getElementById("mask-canvas"),
  dagSvg: document.getElementById("dag-svg"),
  metadataJson: document.getElementById("metadata-json"),
  eventsTable: document.getElementById("events-table"),
  tabs: Array.from(document.querySelectorAll(".tab")),
  tabPanels: Array.from(document.querySelectorAll(".tab-panel")),
};

const palette = [
  "#b55232",
  "#28536b",
  "#7d5a50",
  "#5f0f40",
  "#4f772d",
  "#6f1d1b",
  "#4361ee",
  "#2a9d8f",
  "#8c5e58",
  "#f4a261",
];

async function init() {
  const response = await fetch("/api/bootstrap");
  const payload = await response.json();
  state.defaults = payload.defaults;
  state.config = deepClone(payload.defaults);
  state.ui = payload.ui;
  state.configVersion = 1;
  bindEvents();
  renderForm();
  clearPreview();
  updateToolbarState();
}

function bindEvents() {
  dom.restore.addEventListener("click", () => {
    replaceConfig(deepClone(state.defaults));
    setStatus("Defaults restored.", "ok");
  });

  dom.randomize.addEventListener("click", async () => {
    if (!flushActiveNumericInput()) {
      return;
    }
    const requestId = ++state.requestIds.randomize;
    const startedVersion = state.configVersion;
    setPending("randomize", true);
    setStatus("Randomizing all parameters...", "ok");
    try {
      const response = await fetch("/api/randomize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ seed: state.config.seed ?? undefined }),
      });
      const payload = await response.json();
      if (!response.ok) {
        setStatus(payload.error ?? "Randomization failed.", "error");
        return;
      }
      if (requestId !== state.requestIds.randomize || state.configVersion !== startedVersion) {
        return;
      }
      replaceConfig(payload.config);
      setStatus("All parameters were filled with a valid randomized config.", "ok");
    } catch (error) {
      if (requestId !== state.requestIds.randomize) {
        return;
      }
      setStatus(error.message ?? "Randomization failed.", "error");
    } finally {
      if (requestId === state.requestIds.randomize) {
        setPending("randomize", false);
      }
    }
  });

  dom.preview.addEventListener("click", async () => {
    if (!flushActiveNumericInput()) {
      return;
    }
    const requestId = ++state.requestIds.preview;
    const startedVersion = state.configVersion;
    const requestConfig = deepClone(state.config);
    setPending("preview", true);
    setStatus("Generating in-memory preview...", "ok");
    try {
      const response = await fetch("/api/preview", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ config: requestConfig }),
      });
      const payload = await response.json();
      if (!response.ok) {
        setStatus(payload.error ?? "Preview failed.", "error");
        return;
      }
      if (requestId !== state.requestIds.preview || state.configVersion !== startedVersion) {
        return;
      }
      state.preview = payload.preview;
      const featureCount = state.preview.summary.num_features;
      const maxNodes = Number(dom.maxNodesSelect.value);
      state.selectedNodes = Array.from({ length: Math.min(featureCount, maxNodes) }, (_, index) => index);
      renderPreview();
      setStatus("Preview updated.", "ok");
    } catch (error) {
      if (requestId !== state.requestIds.preview) {
        return;
      }
      setStatus(error.message ?? "Preview failed.", "error");
    } finally {
      if (requestId === state.requestIds.preview) {
        setPending("preview", false);
      }
    }
  });

  dom.datasetSelect.addEventListener("change", renderSeries);
  dom.maxNodesSelect.addEventListener("change", () => {
    if (!state.preview) {
      return;
    }
    const featureCount = state.preview.summary.num_features;
    const maxNodes = Number(dom.maxNodesSelect.value);
    state.selectedNodes = Array.from({ length: Math.min(featureCount, maxNodes) }, (_, index) => index);
    renderNodeSelector();
    renderSeries();
    renderMask();
    renderDag();
  });

  dom.tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      state.activeTab = tab.dataset.tab;
      dom.tabs.forEach((item) => item.classList.toggle("active", item === tab));
      dom.tabPanels.forEach((panel) => panel.classList.toggle("active", panel.id === `tab-${state.activeTab}`));
    });
  });
}

function renderForm() {
  dom.form.innerHTML = "";
  dom.form.appendChild(renderNode(state.config, "", 0));
}

function renderNode(value, path, depth) {
  if (Array.isArray(value)) {
    return renderArrayField(value, path);
  }

  if (isPlainObject(value)) {
    const details = document.createElement("details");
    details.className = "section";
    if (depth < 2 || path === "") {
      details.open = true;
    }

    const summary = document.createElement("summary");
    summary.innerHTML = `<span>${getLabel(path || "root")}</span><span class="field-path">${path || "root"}</span>`;
    details.appendChild(summary);

    const body = document.createElement("div");
    body.className = "section-body field-grid";

    Object.entries(value).forEach(([key, child]) => {
      const childPath = path ? `${path}.${key}` : key;
      if (isLeaf(child)) {
        body.appendChild(renderLeafField(child, childPath));
      } else {
        body.appendChild(renderNode(child, childPath, depth + 1));
      }
    });

    details.appendChild(body);
    return details;
  }

  return renderLeafField(value, path);
}

function renderLeafField(value, path) {
  const wrapper = document.createElement("div");
  wrapper.className = "field";
  wrapper.appendChild(makeLabelBlock(path));

  if (typeof value === "boolean") {
    const checkbox = document.createElement("label");
    checkbox.className = "checkbox-pill";
    checkbox.innerHTML = `<input type="checkbox" ${value ? "checked" : ""} /> Enabled`;
    checkbox.querySelector("input").addEventListener("change", (event) => {
      applyConfigChange(path, event.target.checked);
    });
    wrapper.appendChild(checkbox);
    return wrapper;
  }

  const input = document.createElement("input");
  input.type = "number";
  const numericMeta = state.ui.numericBounds[path] ?? null;
  const numericKind = numericMeta?.kind ?? (Number.isInteger(value) ? "int" : "float");
  input.value = formatNumericValue(value);
  input.step = numericKind === "int" ? "1" : "0.0001";
  input.dataset.path = path;
  input.dataset.kind = numericKind;
  if (numericMeta) {
    input.min = String(numericMeta.min);
    input.max = String(numericMeta.max);
  }
  input.addEventListener("change", () => {
    commitNumericInput(input, path, numericKind);
  });
  wrapper.appendChild(input);
  return wrapper;
}

function renderArrayField(value, path) {
  const wrapper = document.createElement("div");
  wrapper.className = "field";
  wrapper.appendChild(makeLabelBlock(path));

  const options = state.ui.multiSelectOptions[path] ?? value;
  const group = document.createElement("div");
  group.className = "checkbox-group";

  options.forEach((option) => {
    const chip = document.createElement("label");
    chip.className = "checkbox-pill";
    const checked = value.includes(option) ? "checked" : "";
    chip.innerHTML = `<input type="checkbox" value="${option}" ${checked} /> ${option}`;
    chip.querySelector("input").addEventListener("change", () => {
      const current = new Set(getAtPath(state.config, path));
      if (current.has(option)) {
        current.delete(option);
      } else {
        current.add(option);
      }
      if (current.size === 0) {
        current.add(option);
        chip.querySelector("input").checked = true;
      }
      applyConfigChange(path, Array.from(current).sort());
    });
    group.appendChild(chip);
  });

  wrapper.appendChild(group);
  return wrapper;
}

function makeLabelBlock(path) {
  const block = document.createElement("div");
  const label = document.createElement("div");
  label.className = "field-label";
  label.textContent = getLabel(path);
  const pathLabel = document.createElement("div");
  pathLabel.className = "field-path";
  pathLabel.textContent = path;
  block.appendChild(label);
  block.appendChild(pathLabel);
  return block;
}

function renderPreview() {
  if (!state.preview) {
    return;
  }
  renderSummary();
  renderNodeSelector();
  renderSeries();
  renderMask();
  renderDag();
  renderMetadata();
}

function renderSummary() {
  const summary = state.preview.summary;
  dom.summaryChips.innerHTML = "";
  [
    `Length ${summary.length}`,
    `Features ${summary.num_features}`,
    `Events ${summary.num_events}`,
    `Anomalous ${summary.is_anomalous_sample ? "Yes" : "No"}`,
  ].forEach((text) => {
    const chip = document.createElement("div");
    chip.className = "chip";
    chip.textContent = text;
    dom.summaryChips.appendChild(chip);
  });
}

function renderNodeSelector() {
  if (!state.preview) {
    return;
  }
  dom.nodeSelector.innerHTML = "";
  const featureCount = state.preview.summary.num_features;
  for (let index = 0; index < featureCount; index += 1) {
    const chip = document.createElement("label");
    chip.className = "checkbox-pill";
    chip.innerHTML = `<input type="checkbox" value="${index}" ${state.selectedNodes.includes(index) ? "checked" : ""} /> node ${index}`;
    chip.querySelector("input").addEventListener("change", (event) => {
      const node = Number(event.target.value);
      if (event.target.checked) {
        state.selectedNodes = [...state.selectedNodes, node].sort((a, b) => a - b);
      } else {
        state.selectedNodes = state.selectedNodes.filter((value) => value !== node);
        if (state.selectedNodes.length === 0) {
          state.selectedNodes = [node];
          event.target.checked = true;
        }
      }
      renderSeries();
      renderMask();
      renderDag();
    });
    dom.nodeSelector.appendChild(chip);
  }
}

function renderSeries() {
  if (!state.preview) {
    clearCanvas(dom.seriesCanvas);
    return;
  }
  const datasetName = dom.datasetSelect.value;
  const series = state.preview.series[datasetName];
  const mask = state.preview.labels.point_mask_any;
  drawLineChart(dom.seriesCanvas, series, state.selectedNodes, mask);
}

function renderMask() {
  if (!state.preview) {
    clearCanvas(dom.maskCanvas);
    return;
  }
  drawMaskHeatmap(dom.maskCanvas, state.preview.labels.point_mask, state.selectedNodes);
}

function renderDag() {
  if (!state.preview) {
    dom.dagSvg.innerHTML = "";
    return;
  }
  drawDag(dom.dagSvg, state.preview.graph.parents, state.preview.graph.topo_order, state.selectedNodes);
}

function renderMetadata() {
  dom.metadataJson.textContent = JSON.stringify(
    {
      metadata: state.preview.metadata,
      graph: state.preview.graph,
      labels: {
        root_cause: state.preview.labels.root_cause,
        affected_nodes: state.preview.labels.affected_nodes,
      },
    },
    null,
    2,
  );

  const events = state.preview.labels.events ?? [];
  dom.eventsTable.innerHTML = "";
  if (events.length === 0) {
    dom.eventsTable.innerHTML = '<p class="subtle">No realized events for this preview.</p>';
    return;
  }

  events.forEach((event) => {
    const card = document.createElement("div");
    card.className = "event-card";
    card.innerHTML = `
      <strong>${event.anomaly_type}</strong>
      <div>node ${event.node}, [${event.t_start}, ${event.t_end})</div>
      <div>affected: ${(event.affected_nodes ?? []).join(", ") || "-"}</div>
      <div>root cause: ${event.root_cause_node ?? "-"}</div>
    `;
    dom.eventsTable.appendChild(card);
  });
}

function drawLineChart(canvas, data, selectedNodes, maskAny) {
  const context = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#ffffff";
  context.fillRect(0, 0, width, height);

  if (!data || selectedNodes.length === 0) {
    return;
  }

  const padding = { left: 60, right: 24, top: 24, bottom: 40 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const values = selectedNodes.flatMap((node) => data.map((row) => row[node]));
  let minValue = Math.min(...values);
  let maxValue = Math.max(...values);
  if (Math.abs(maxValue - minValue) < 1e-8) {
    minValue -= 1;
    maxValue += 1;
  }

  context.fillStyle = "rgba(181, 82, 50, 0.08)";
  let inMask = false;
  let maskStart = 0;
  maskAny.forEach((value, index) => {
    if (value && !inMask) {
      inMask = true;
      maskStart = index;
    }
    const shouldClose = !value && inMask;
    const isLast = inMask && index === maskAny.length - 1;
    if (shouldClose || isLast) {
      const maskEnd = shouldClose ? index : index + 1;
      const x1 = padding.left + (maskStart / Math.max(1, data.length - 1)) * plotWidth;
      const x2 = padding.left + ((maskEnd - 1) / Math.max(1, data.length - 1)) * plotWidth;
      context.fillRect(x1, padding.top, Math.max(2, x2 - x1), plotHeight);
      inMask = false;
    }
  });

  drawAxes(context, padding, width, height, minValue, maxValue, data.length);

  selectedNodes.forEach((node, index) => {
    context.strokeStyle = palette[index % palette.length];
    context.lineWidth = 2;
    context.beginPath();
    data.forEach((row, step) => {
      const x = padding.left + (step / Math.max(1, data.length - 1)) * plotWidth;
      const ratio = (row[node] - minValue) / (maxValue - minValue);
      const y = padding.top + plotHeight - ratio * plotHeight;
      if (step === 0) {
        context.moveTo(x, y);
      } else {
        context.lineTo(x, y);
      }
    });
    context.stroke();
  });

  context.font = "12px Segoe UI";
  context.fillStyle = "#1e1a16";
  selectedNodes.forEach((node, index) => {
    const legendX = padding.left + index * 110;
    context.fillStyle = palette[index % palette.length];
    context.fillRect(legendX, 8, 16, 4);
    context.fillStyle = "#1e1a16";
    context.fillText(`node ${node}`, legendX + 22, 14);
  });
}

function drawAxes(context, padding, width, height, minValue, maxValue, length) {
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  context.strokeStyle = "#bcb3a6";
  context.lineWidth = 1;
  context.beginPath();
  context.moveTo(padding.left, padding.top);
  context.lineTo(padding.left, padding.top + plotHeight);
  context.lineTo(padding.left + plotWidth, padding.top + plotHeight);
  context.stroke();

  context.fillStyle = "#6f6559";
  context.font = "12px Segoe UI";
  for (let tick = 0; tick <= 4; tick += 1) {
    const ratio = tick / 4;
    const y = padding.top + plotHeight - ratio * plotHeight;
    const value = minValue + ratio * (maxValue - minValue);
    context.fillText(value.toFixed(2), 6, y + 4);
    context.strokeStyle = "rgba(111, 101, 89, 0.15)";
    context.beginPath();
    context.moveTo(padding.left, y);
    context.lineTo(padding.left + plotWidth, y);
    context.stroke();
  }
  context.fillText("t=0", padding.left, height - 10);
  context.fillText(`t=${length - 1}`, width - 70, height - 10);
}

function drawMaskHeatmap(canvas, pointMask, selectedNodes) {
  const context = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#ffffff";
  context.fillRect(0, 0, width, height);

  if (!pointMask || selectedNodes.length === 0) {
    return;
  }

  const padding = { left: 90, right: 20, top: 20, bottom: 24 };
  const rows = selectedNodes.length;
  const columns = pointMask.length;
  const cellWidth = (width - padding.left - padding.right) / Math.max(1, columns);
  const cellHeight = (height - padding.top - padding.bottom) / Math.max(1, rows);

  selectedNodes.forEach((node, rowIndex) => {
    context.fillStyle = "#1e1a16";
    context.fillText(`node ${node}`, 16, padding.top + rowIndex * cellHeight + cellHeight * 0.7);
    for (let column = 0; column < columns; column += 1) {
      context.fillStyle = pointMask[column][node] ? "#b55232" : "#f3ebe0";
      context.fillRect(
        padding.left + column * cellWidth,
        padding.top + rowIndex * cellHeight,
        Math.max(1, cellWidth),
        Math.max(1, cellHeight - 1),
      );
    }
  });
}

function drawDag(svg, parents, topoOrder, selectedNodes) {
  svg.innerHTML = "";
  if (!parents) {
    return;
  }

  const width = 980;
  const height = 420;
  const depths = Array.from({ length: parents.length }, () => 0);
  topoOrder.forEach((node) => {
    if (parents[node].length > 0) {
      depths[node] = Math.max(...parents[node].map((parent) => depths[parent] + 1));
    }
  });

  const levels = new Map();
  topoOrder.forEach((node) => {
    const depth = depths[node];
    if (!levels.has(depth)) {
      levels.set(depth, []);
    }
    levels.get(depth).push(node);
  });

  const maxDepth = Math.max(...levels.keys(), 0);
  const positions = new Map();
  levels.forEach((nodes, depth) => {
    nodes.forEach((node, index) => {
      const x = maxDepth === 0 ? width / 2 : 90 + (depth / maxDepth) * (width - 180);
      const y = nodes.length === 1 ? height / 2 : 60 + (index / (nodes.length - 1)) * (height - 120);
      positions.set(node, { x, y });
    });
  });

  const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
  defs.innerHTML = `
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#5a554e"></path>
    </marker>
  `;
  svg.appendChild(defs);

  parents.forEach((nodeParents, child) => {
    nodeParents.forEach((parent) => {
      const from = positions.get(parent);
      const to = positions.get(child);
      const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
      line.setAttribute("x1", from.x + 22);
      line.setAttribute("y1", from.y);
      line.setAttribute("x2", to.x - 22);
      line.setAttribute("y2", to.y);
      line.setAttribute("stroke", "#5a554e");
      line.setAttribute("stroke-width", "2");
      line.setAttribute("marker-end", "url(#arrow)");
      svg.appendChild(line);
    });
  });

  positions.forEach(({ x, y }, node) => {
    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("cx", x);
    circle.setAttribute("cy", y);
    circle.setAttribute("r", 20);
    circle.setAttribute("fill", selectedNodes.includes(node) ? "#b55232" : "#28536b");
    circle.setAttribute("stroke", "#1e1a16");
    circle.setAttribute("stroke-width", "1.5");
    svg.appendChild(circle);

    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("x", x);
    text.setAttribute("y", y + 5);
    text.setAttribute("text-anchor", "middle");
    text.setAttribute("fill", "#ffffff");
    text.setAttribute("font-size", "12");
    text.setAttribute("font-weight", "700");
    text.textContent = String(node);
    svg.appendChild(text);
  });
}

function clearCanvas(canvas) {
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = "#ffffff";
  context.fillRect(0, 0, canvas.width, canvas.height);
}

function clearPreview() {
  state.preview = null;
  state.selectedNodes = [];
  dom.summaryChips.innerHTML = "";
  dom.nodeSelector.innerHTML = "";
  clearCanvas(dom.seriesCanvas);
  clearCanvas(dom.maskCanvas);
  dom.dagSvg.innerHTML = "";
  dom.metadataJson.textContent = "";
  dom.eventsTable.innerHTML = '<p class="subtle">Generate a preview to inspect events and metadata.</p>';
}

function replaceConfig(nextConfig) {
  state.config = nextConfig;
  state.configVersion += 1;
  renderForm();
  clearPreview();
  hideStatus();
}

function applyConfigChange(path, value) {
  setAtPath(state.config, path, value);
  state.configVersion += 1;
  clearPreview();
  hideStatus();
}

function commitNumericInput(input, path, kind) {
  const rawValue = input.value.trim();
  if (rawValue === "") {
    input.value = formatNumericValue(getAtPath(state.config, path));
    setStatus(`Value required for ${path}.`, "error");
    return false;
  }

  const nextValue = Number(rawValue);
  const error = validateNumericValue(path, nextValue, kind);
  if (error) {
    input.value = formatNumericValue(getAtPath(state.config, path));
    setStatus(error, "error");
    return false;
  }

  const currentValue = getAtPath(state.config, path);
  input.value = formatNumericValue(nextValue);
  if (Object.is(currentValue, nextValue)) {
    clearErrorStatus();
    return true;
  }

  applyConfigChange(path, nextValue);
  return true;
}

function validateNumericValue(path, value, kind) {
  if (!Number.isFinite(value)) {
    return `Invalid numeric value for ${path}.`;
  }
  if (kind === "int" && !Number.isInteger(value)) {
    return `Expected an integer for ${path}.`;
  }

  const numericMeta = state.ui.numericBounds[path];
  if (numericMeta && (value < numericMeta.min || value > numericMeta.max)) {
    return `${path} must be between ${numericMeta.min} and ${numericMeta.max}.`;
  }

  if (path.endsWith(".min")) {
    const basePath = path.slice(0, -4);
    if (value > getAtPath(state.config, `${basePath}.max`)) {
      return `Min cannot exceed max for ${basePath}.`;
    }
  }

  if (path.endsWith(".max")) {
    const basePath = path.slice(0, -4);
    if (value < getAtPath(state.config, `${basePath}.min`)) {
      return `Max cannot be smaller than min for ${basePath}.`;
    }
  }

  return null;
}

function flushActiveNumericInput() {
  const activeElement = document.activeElement;
  if (!(activeElement instanceof HTMLInputElement) || activeElement.type !== "number") {
    return true;
  }
  const path = activeElement.dataset.path;
  const kind = activeElement.dataset.kind ?? "float";
  if (!path) {
    return true;
  }
  return commitNumericInput(activeElement, path, kind);
}

function formatNumericValue(value) {
  return String(value);
}

function setPending(action, value) {
  state.pending[action] = value;
  updateToolbarState();
}

function updateToolbarState() {
  const busy = state.pending.randomize || state.pending.preview;
  dom.restore.disabled = busy;
  dom.randomize.disabled = busy;
  dom.preview.disabled = busy;
}

function setStatus(message, kind) {
  dom.status.textContent = message;
  dom.status.classList.remove("hidden", "error", "ok");
  dom.status.classList.add(kind);
}

function hideStatus() {
  dom.status.textContent = "";
  dom.status.classList.add("hidden");
  dom.status.classList.remove("error", "ok");
}

function clearErrorStatus() {
  if (dom.status.classList.contains("error")) {
    hideStatus();
  }
}

function getLabel(path) {
  if (!path) {
    return state.ui.pathLabels.root ?? "Root";
  }
  return state.ui.pathLabels[path] ?? path.split(".").slice(-1)[0];
}

function isLeaf(value) {
  return Array.isArray(value) || !isPlainObject(value);
}

function isPlainObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function setAtPath(target, path, value) {
  const parts = path.split(".");
  let current = target;
  for (let index = 0; index < parts.length - 1; index += 1) {
    current = current[parts[index]];
  }
  current[parts[parts.length - 1]] = value;
}

function getAtPath(target, path) {
  return path.split(".").reduce((current, key) => current[key], target);
}

function deepClone(value) {
  return JSON.parse(JSON.stringify(value));
}

init().catch((error) => {
  console.error(error);
  setStatus(error.message ?? "Failed to load studio.", "error");
});
