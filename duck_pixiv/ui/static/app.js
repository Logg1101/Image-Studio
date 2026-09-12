// Duck Pixiv Assistant - Frontend Logic
let selectedImages = [];
let currentPostData = null;
let currentBrowserPath = "";
let currentParentPath = null;

// Initialize on page load
document.addEventListener("DOMContentLoaded", () => {
  loadFolderContent(); // Start at outputs folder
  loadSettingsConfig();
});

// Mode Switching: Folder Browser vs Manual File Upload
function switchSelectorMode(mode) {
  const tabFolderBtn = document.getElementById("tabFolderBtn");
  const tabUploadBtn = document.getElementById("tabUploadBtn");
  const folderView = document.getElementById("folderBrowserView");
  const uploadView = document.getElementById("uploadFilesView");

  if (mode === "folder") {
    tabFolderBtn.className = "px-3 py-1 rounded bg-blue-600 text-white font-medium flex items-center gap-1.5 transition";
    tabUploadBtn.className = "px-3 py-1 rounded bg-[#21262d] text-gray-400 hover:text-white font-medium flex items-center gap-1.5 transition";
    folderView.classList.remove("hidden");
    uploadView.classList.add("hidden");
  } else {
    tabUploadBtn.className = "px-3 py-1 rounded bg-blue-600 text-white font-medium flex items-center gap-1.5 transition";
    tabFolderBtn.className = "px-3 py-1 rounded bg-[#21262d] text-gray-400 hover:text-white font-medium flex items-center gap-1.5 transition";
    uploadView.classList.remove("hidden");
    folderView.classList.add("hidden");
  }
}

// Load folder contents (subfolders and image preview thumbnails)
async function loadFolderContent(folderPath = null) {
  const pathDisplay = document.getElementById("currentFolderPath");
  const subfoldersList = document.getElementById("subfoldersList");
  const imagesGrid = document.getElementById("folderImagesGrid");
  const upBtn = document.getElementById("folderUpBtn");

  imagesGrid.innerHTML = '<div class="text-xs text-gray-500 col-span-4 py-8 text-center"><i class="fa-solid fa-spinner fa-spin mr-1"></i> Loading folder...</div>';

  try {
    const url = folderPath ? `/api/explore_folder?folder_path=${encodeURIComponent(folderPath)}` : `/api/explore_folder`;
    const res = await fetch(url);
    const data = await res.json();

    currentBrowserPath = data.current_path;
    currentParentPath = data.parent_path;

    // Display formatted relative path
    const parts = currentBrowserPath.replace(/\\/g, "/").split("/");
    const shortPath = parts.slice(-3).join("/");
    pathDisplay.textContent = shortPath || currentBrowserPath;
    pathDisplay.title = currentBrowserPath;

    // Parent navigation button state
    upBtn.disabled = !currentParentPath;

    // Render subfolders
    document.getElementById("subfolderCount").textContent = `${data.subfolders.length} folders`;
    if (!data.subfolders || data.subfolders.length === 0) {
      subfoldersList.innerHTML = '<span class="text-[11px] text-gray-500 italic py-1">No subfolders</span>';
    } else {
      subfoldersList.innerHTML = "";
      data.subfolders.forEach(f => {
        const btn = document.createElement("button");
        btn.className = "px-2 py-1 rounded bg-[#21262d] hover:bg-blue-600 hover:text-white text-gray-300 text-[11px] flex items-center gap-1 transition border border-[#30363d]";
        btn.title = f.path;
        btn.innerHTML = `<i class="fa-regular fa-folder text-blue-400"></i> <span class="truncate max-w-[120px]">${f.name}</span>`;
        btn.onclick = () => loadFolderContent(f.path);
        subfoldersList.appendChild(btn);
      });
    }

    // Render image previews
    document.getElementById("folderImageCount").textContent = `${data.images.length} images`;
    if (!data.images || data.images.length === 0) {
      imagesGrid.innerHTML = '<div class="text-xs text-gray-500 col-span-4 py-8 text-center">No images found in this folder</div>';
      return;
    }

    imagesGrid.innerHTML = "";
    data.images.forEach(img => {
      const isAlreadySelected = selectedImages.some(s => s.path === img.path);
      const div = document.createElement("div");
      div.className = `relative group cursor-pointer aspect-square rounded overflow-hidden border transition ${
        isAlreadySelected ? 'border-blue-500 ring-2 ring-blue-500/40' : 'border-[#30363d] hover:border-blue-400'
      }`;
      div.title = img.name;
      div.onclick = () => toggleImageSelection(img.path, img.name);
      div.innerHTML = `
        <img src="/api/image_file?path=${encodeURIComponent(img.path)}" class="w-full h-full object-cover group-hover:scale-105 transition" loading="lazy">
        <div class="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 flex flex-col items-center justify-center transition text-white text-[11px] font-bold p-1 text-center">
          <span>${isAlreadySelected ? 'Remove' : 'Select'}</span>
        </div>
        ${isAlreadySelected ? '<div class="absolute top-1 right-1 w-4 h-4 bg-blue-600 rounded-full flex items-center justify-center text-white text-[9px] shadow"><i class="fa-solid fa-check"></i></div>' : ''}
      `;
      imagesGrid.appendChild(div);
    });

  } catch (err) {
    imagesGrid.innerHTML = `<div class="text-xs text-red-400 col-span-4 py-8 text-center">Failed to load folder: ${err.message}</div>`;
  }
}

function navigateFolderUp() {
  if (currentParentPath) {
    loadFolderContent(currentParentPath);
  }
}

function refreshCurrentFolder() {
  loadFolderContent(currentBrowserPath);
}

function toggleImageSelection(path, name) {
  const existingIdx = selectedImages.findIndex(i => i.path === path);
  if (existingIdx >= 0) {
    selectedImages.splice(existingIdx, 1);
    if (activePreviewIndex >= selectedImages.length) {
      activePreviewIndex = Math.max(0, selectedImages.length - 1);
    }
  } else {
    selectedImages.push({ path: path, name: name || path.split(/[/\\]/).pop() });
    activePreviewIndex = selectedImages.length - 1; // Preview newly added image
  }
  updateSelectedUI();
  refreshCurrentFolder(); // Refresh checkmarks on grid
  if (selectedImages.length > 0) {
    setLargePreview(activePreviewIndex);
    processPostPreparation();
  }
}

// Add an image path to selection
function addImagePath(path, name) {
  if (selectedImages.some(i => i.path === path)) {
    showBanner("This image is already selected.", "warning");
    return;
  }
  selectedImages.push({ path: path, name: name || path.split(/[/\\]/).pop() });
  activePreviewIndex = selectedImages.length - 1;
  updateSelectedUI();
  setLargePreview(activePreviewIndex);
  processPostPreparation();
}

// Handle file input select (local upload)
async function handleFileSelect(event) {
  const files = event.target.files;
  if (!files || files.length === 0) return;

  const formData = new FormData();
  for (let i = 0; i < files.length; i++) {
    formData.append("files", files[i]);
  }

  showBanner("Uploading & analyzing images...", "info");
  try {
    const res = await fetch("/api/upload_local_files", {
      method: "POST",
      body: formData
    });
    const data = await res.json();
    if (data.paths && data.paths.length > 0) {
      data.paths.forEach(p => {
        if (!selectedImages.some(i => i.path === p)) {
          selectedImages.push({ path: p, name: p.split(/[/\\]/).pop() });
        }
      });
      updateSelectedUI();
      processPostPreparation();
      showBanner(`Loaded ${data.paths.length} image(s).`, "success");
    }
  } catch (err) {
    showBanner(`Failed to load image files: ${err.message}`, "error");
  }
}

let activePreviewIndex = 0;

// Set the active artwork in the large preview pane
function setLargePreview(index) {
  if (index < 0 || index >= selectedImages.length) return;
  activePreviewIndex = index;

  const img = selectedImages[index];
  const previewImg = document.getElementById("largePreviewImg");
  const placeholder = document.getElementById("previewPlaceholder");
  const overlay = document.getElementById("previewOverlay");
  const fullLink = document.getElementById("previewFullSizeLink");
  const fileNameLabel = document.getElementById("previewFileName");
  const indexLabel = document.getElementById("previewIndexIndicator");
  const resBadge = document.getElementById("previewResolutionBadge");

  const imgSrc = `/api/image_file?path=${encodeURIComponent(img.path)}`;

  previewImg.src = imgSrc;
  previewImg.classList.remove("hidden");
  placeholder.classList.add("hidden");
  overlay.classList.remove("hidden");
  fullLink.href = imgSrc;
  fileNameLabel.textContent = img.name;
  fileNameLabel.title = img.path;
  indexLabel.textContent = `Image ${index + 1} of ${selectedImages.length}`;

  // Read natural resolution once loaded
  previewImg.onload = () => {
    resBadge.textContent = `${previewImg.naturalWidth} × ${previewImg.naturalHeight}`;
  };

  // Update selection highlight in list
  updateSelectedUI();
}

// Update selected images list and order
function updateSelectedUI() {
  const container = document.getElementById("selectedImagesContainer");
  const countBadge = document.getElementById("selectedCount");
  countBadge.textContent = `${selectedImages.length} Selected`;

  if (selectedImages.length === 0) {
    container.innerHTML = `
      <div class="flex flex-col items-center justify-center text-gray-500 py-8">
        <span class="text-xs">No images selected</span>
      </div>
    `;
    // Reset preview pane
    document.getElementById("largePreviewImg").classList.add("hidden");
    document.getElementById("previewPlaceholder").classList.remove("hidden");
    document.getElementById("previewOverlay").classList.add("hidden");
    document.getElementById("previewFileName").textContent = "--";
    document.getElementById("previewIndexIndicator").textContent = "Image 0 of 0";
    document.getElementById("previewResolutionBadge").textContent = "--";
    resetForm();
    return;
  }

  // Ensure activePreviewIndex is within range
  if (activePreviewIndex >= selectedImages.length) {
    activePreviewIndex = selectedImages.length - 1;
  }

  container.innerHTML = "";
  selectedImages.forEach((img, index) => {
    const isCurrentPreview = index === activePreviewIndex;
    const div = document.createElement("div");
    div.className = `flex items-center gap-3 p-2 rounded-lg border cursor-pointer transition ${
      isCurrentPreview ? 'bg-blue-950/40 border-blue-500' : 'bg-[#0d1117] border-[#30363d] hover:border-gray-600'
    }`;
    div.onclick = (e) => {
      // Don't trigger if clicked on buttons
      if (e.target.closest("button")) return;
      setLargePreview(index);
    };
    div.innerHTML = `
      <span class="text-xs font-mono ${isCurrentPreview ? 'text-blue-400 font-bold' : 'text-gray-500'} w-4 text-center">${index + 1}</span>
      <img src="/api/image_file?path=${encodeURIComponent(img.path)}" class="w-11 h-11 object-cover rounded border ${isCurrentPreview ? 'border-blue-400' : 'border-gray-700'}">
      <div class="flex-1 min-w-0">
        <p class="text-xs font-medium ${isCurrentPreview ? 'text-white' : 'text-gray-200'} truncate">${img.name}</p>
        <p class="text-[10px] text-gray-500 truncate">${img.path}</p>
      </div>
      <div class="flex items-center gap-1 text-gray-400">
        <button onclick="moveImage(${index}, -1)" class="hover:text-blue-400 p-1 text-xs" title="Move Up" ${index === 0 ? 'disabled' : ''}>
          <i class="fa-solid fa-arrow-up"></i>
        </button>
        <button onclick="moveImage(${index}, 1)" class="hover:text-blue-400 p-1 text-xs" title="Move Down" ${index === selectedImages.length - 1 ? 'disabled' : ''}>
          <i class="fa-solid fa-arrow-down"></i>
        </button>
        <button onclick="removeImage(${index})" class="hover:text-red-400 p-1 text-xs" title="Remove">
          <i class="fa-solid fa-trash-can"></i>
        </button>
      </div>
    `;
    container.appendChild(div);
  });
}

function moveImage(index, dir) {
  const target = index + dir;
  if (target < 0 || target >= selectedImages.length) return;
  const temp = selectedImages[index];
  selectedImages[index] = selectedImages[target];
  selectedImages[target] = temp;
  updateSelectedUI();
  processPostPreparation();
}

function removeImage(index) {
  selectedImages.splice(index, 1);
  updateSelectedUI();
  if (selectedImages.length > 0) {
    processPostPreparation();
  }
}

function clearSelectedImages() {
  selectedImages = [];
  updateSelectedUI();
}

// Request preparation & metadata extraction
async function processPostPreparation() {
  if (selectedImages.length === 0) return;

  showBanner("Analyzing generation metadata...", "info");
  try {
    const paths = selectedImages.map(i => i.path);
    const res = await fetch("/api/prepare_post", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_paths: paths })
    });

    if (!res.ok) {
      const errData = await res.json();
      throw new Error(errData.detail || "Analysis error");
    }

    currentPostData = await res.json();
    renderExtractedData(currentPostData);
    showBanner("Metadata extracted and converted successfully.", "success");
  } catch (err) {
    showBanner(`Analysis failed: ${err.message}`, "error");
  }
}

// Render metadata breakdown, titles, description, tags
function renderExtractedData(data) {
  const pmeta = data.primary_meta || {};
  document.getElementById("modelBadge").textContent = pmeta.model || "SDXL";

  const feats = data.extracted_features || {};
  const charJp = feats.character && feats.character.length ? feats.character.join(", ") : (pmeta.character_detected || "Not specified");
  document.getElementById("metaCharacter").textContent = charJp;
  document.getElementById("metaClothing").textContent = feats.clothing ? feats.clothing.slice(0, 4).join(", ") : "--";
  document.getElementById("metaAppearance").textContent = feats.appearance ? feats.appearance.slice(0, 4).join(", ") : "--";
  document.getElementById("metaPose").textContent = feats.pose ? feats.pose.slice(0, 3).join(", ") : "--";
  document.getElementById("metaExpression").textContent = feats.expression ? feats.expression.slice(0, 3).join(", ") : "--";
  document.getElementById("metaMaterials").textContent = feats.materials ? feats.materials.slice(0, 3).join(", ") : "--";

  // Titles
  renderTitleCandidates(data.title_candidates || [], data.selected_title || "");

  // Description & English translation box
  document.getElementById("descriptionInput").value = data.description || "";
  document.getElementById("descriptionEnInput").value = data.description_en || "";

  // Tags
  renderTags(data.tags || []);
}

function renderTitleCandidates(titles, selected) {
  const container = document.getElementById("titleCandidatesList");
  container.innerHTML = "";
  
  let selectedEn = "";

  titles.forEach((item, idx) => {
    // item can be an object {ja: "...", en: "..."} or a legacy string
    const titleJa = typeof item === "object" ? item.ja : item;
    const titleEn = typeof item === "object" ? item.en : "";

    const isSel = titleJa === selected;
    if (isSel && titleEn) {
      selectedEn = titleEn;
    }

    const div = document.createElement("div");
    div.className = `flex items-center justify-between p-3 rounded-lg border text-xs cursor-pointer transition ${
      isSel ? "border-blue-500 bg-blue-950/30 text-white font-semibold" : "border-[#30363d] bg-[#0d1117] text-gray-300 hover:border-gray-500"
    }`;
    div.onclick = () => selectTitle(titleJa, titleEn);
    div.innerHTML = `
      <div class="flex items-start gap-2.5 flex-1 pr-2">
        <span class="w-5 h-5 rounded-full flex items-center justify-center text-[10px] mt-0.5 shrink-0 ${isSel ? 'bg-blue-600 text-white' : 'bg-gray-800 text-gray-400'}">
          ${idx + 1}
        </span>
        <div class="flex-1">
          <div class="text-white text-sm font-medium leading-tight">${titleJa}</div>
          ${titleEn ? `<div class="text-[11px] text-blue-300/80 mt-0.5 italic flex items-center gap-1"><i class="fa-solid fa-language text-[10px] opacity-70"></i> ${titleEn}</div>` : ''}
        </div>
      </div>
      <button class="px-2.5 py-1 rounded text-[11px] font-medium shrink-0 ${isSel ? 'bg-blue-600 text-white' : 'bg-[#21262d] text-gray-400 hover:text-white'}">
        ${isSel ? 'Selected' : 'Apply'}
      </button>
    `;
    container.appendChild(div);
  });

  document.getElementById("finalTitleInput").value = selected;
  document.getElementById("finalTitleEnPreview").textContent = selectedEn || "--";
}

function selectTitle(titleJa, titleEn) {
  if (currentPostData) {
    currentPostData.selected_title = titleJa;
  }
  document.getElementById("finalTitleInput").value = titleJa;
  document.getElementById("finalTitleEnPreview").textContent = titleEn || "--";
  renderTitleCandidates(currentPostData ? currentPostData.title_candidates : [], titleJa);
}

// Tag chips rendering
function renderTags(tags) {
  const container = document.getElementById("tagsContainer");
  const countBadge = document.getElementById("tagCountBadge");
  countBadge.textContent = `${tags.length} / 10`;
  countBadge.className = `text-xs px-2 py-0.5 rounded border ${
    tags.length > 10 ? 'bg-red-950 text-red-400 border-red-800' : 'bg-gray-800 text-blue-400 border-[#30363d]'
  }`;

  if (!tags || tags.length === 0) {
    container.innerHTML = '<span class="text-xs text-gray-500 italic p-1">No tags available</span>';
    return;
  }

  container.innerHTML = "";
  tags.forEach((tag, idx) => {
    const span = document.createElement("span");
    span.className = "tag-chip px-2.5 py-1 rounded-md text-xs text-blue-200 flex items-center gap-1.5 font-medium";
    span.innerHTML = `
      ${tag}
      <button onclick="removeTag(${idx})" class="text-gray-400 hover:text-red-400 text-[10px] ml-0.5" title="Remove tag">
        <i class="fa-solid fa-xmark"></i>
      </button>
    `;
    container.appendChild(span);
  });
}

function removeTag(idx) {
  if (!currentPostData || !currentPostData.tags) return;
  currentPostData.tags.splice(idx, 1);
  renderTags(currentPostData.tags);
}

function addCustomTag() {
  const input = document.getElementById("newTagInput");
  const tag = input.value.trim();
  if (!tag) return;
  if (!currentPostData) currentPostData = { tags: [] };
  if (!currentPostData.tags) currentPostData.tags = [];

  if (currentPostData.tags.includes(tag)) {
    showBanner("Tag already added.", "warning");
    return;
  }
  if (currentPostData.tags.length >= 10) {
    showBanner("Pixiv limits tags to 10 maximum.", "warning");
    return;
  }

  currentPostData.tags.push(tag);
  renderTags(currentPostData.tags);
  input.value = "";
}

async function regenerateTitles() {
  if (!currentPostData) return;
  try {
    const res = await fetch("/api/regenerate_titles", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ metadata: currentPostData.primary_meta })
    });
    const data = await res.json();
    currentPostData.title_candidates = data.titles;
    const firstTitleJa = typeof data.titles[0] === "object" ? data.titles[0].ja : data.titles[0];
    currentPostData.selected_title = firstTitleJa;
    renderTitleCandidates(data.titles, firstTitleJa);
    showBanner("Regenerated title candidates.", "success");
  } catch (err) {
    showBanner(`Title generation error: ${err.message}`, "error");
  }
}

async function regenerateDescription() {
  if (!currentPostData) return;
  try {
    const res = await fetch("/api/regenerate_description", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ 
        metadata: currentPostData.primary_meta,
        selected_title: document.getElementById("finalTitleInput").value 
      })
    });
    const data = await res.json();
    const descJa = typeof data.description === "object" ? data.description.ja : data.description;
    const descEn = typeof data.description === "object" ? data.description.en : "";

    document.getElementById("descriptionInput").value = descJa;
    document.getElementById("descriptionEnInput").value = descEn;
    currentPostData.description = descJa;
    currentPostData.description_en = descEn;
    showBanner("Regenerated description caption & English translation.", "success");
  } catch (err) {
    showBanner(`Description generation error: ${err.message}`, "error");
  }
}

async function regenerateTags() {
  if (!currentPostData) return;
  try {
    const res = await fetch("/api/regenerate_tags", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ metadata: currentPostData.primary_meta })
    });
    const data = await res.json();
    currentPostData.tags = data.tags;
    renderTags(data.tags);
    showBanner("Regenerated tags from dictionary & fixed tags.", "success");
  } catch (err) {
    showBanner(`Tag generation error: ${err.message}`, "error");
  }
}

// Submit Post
async function submitPost() {
  if (!currentPostData || selectedImages.length === 0) {
    showBanner("Please select image(s) to post.", "warning");
    return;
  }

  const finalTitle = document.getElementById("finalTitleInput").value.trim();
  if (!finalTitle) {
    showBanner("Please enter a title.", "warning");
    return;
  }

  if (currentPostData.tags.length > 10) {
    showBanner("Pixiv limits tags to 10 maximum. Please remove excess tags.", "warning");
    return;
  }

  const submitBtn = document.getElementById("submitPostBtn");
  submitBtn.disabled = true;
  submitBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Posting...';

  // Build payload
  const payload = {
    image_paths: selectedImages.map(i => i.path),
    primary_meta: currentPostData.primary_meta,
    title_candidates: currentPostData.title_candidates,
    selected_title: finalTitle,
    description: document.getElementById("descriptionInput").value,
    tags: currentPostData.tags,
    is_ai: document.getElementById("isAiSelect").value === "true",
    age_limit: document.getElementById("ageLimitSelect").value
  };

  try {
    const res = await fetch("/api/submit_post", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const result = await res.json();

    if (result.success) {
      showBanner(`[Success] ${result.message}`, "success");
      showPostResultModal(true, finalTitle, result.pixiv_id, result.message);
    } else {
      showBanner(`[Failed] ${result.message}`, "error");
      showPostResultModal(false, finalTitle, null, result.message);
    }
  } catch (err) {
    showBanner(`Network error: ${err.message}`, "error");
    showPostResultModal(false, finalTitle, null, err.message);
  } finally {
    submitBtn.disabled = false;
    submitBtn.innerHTML = '<i class="fa-solid fa-paper-plane"></i> 投稿する (Post to Pixiv)';
  }
}

// Show Post Confirmation Modal
function showPostResultModal(success, title, pixivId, message) {
  const modal = document.getElementById("postResultModal");
  const iconBox = document.getElementById("resultIconBox");
  const icon = document.getElementById("resultIcon");
  const titleEl = document.getElementById("resultTitle");
  const subtitleEl = document.getElementById("resultSubtitle");
  const titleVal = document.getElementById("resultTitleVal");
  const idVal = document.getElementById("resultIdVal");
  const modeVal = document.getElementById("resultModeVal");
  const msgVal = document.getElementById("resultMessageVal");
  const pixivLink = document.getElementById("resultPixivLink");

  titleVal.textContent = title;
  msgVal.textContent = message || "--";

  const mode = document.getElementById("uploadModeLabel").textContent.trim();
  modeVal.textContent = mode;

  if (success) {
    iconBox.className = "w-10 h-10 rounded-full bg-green-900/60 text-green-400 border border-green-700 flex items-center justify-center text-lg";
    icon.className = "fa-solid fa-check";
    titleEl.textContent = "Upload Succeeded!";
    subtitleEl.textContent = "Your artwork is prepared & recorded.";
    idVal.textContent = pixivId ? `#${pixivId}` : "Assigned";

    if (pixivId && mode.includes("Pixiv")) {
      pixivLink.href = `https://www.pixiv.net/artworks/${pixivId}`;
      pixivLink.classList.remove("hidden");
    } else {
      pixivLink.classList.add("hidden");
    }
  } else {
    iconBox.className = "w-10 h-10 rounded-full bg-red-900/60 text-red-400 border border-red-700 flex items-center justify-center text-lg";
    icon.className = "fa-solid fa-triangle-exclamation";
    titleEl.textContent = "Upload Failed";
    subtitleEl.textContent = "Please review the error details below.";
    idVal.textContent = "None";
    pixivLink.classList.add("hidden");
  }

  modal.classList.remove("hidden");
}

function closePostResultModal() {
  document.getElementById("postResultModal").classList.add("hidden");
}

// Preview Final Post
function previewFinalPost() {
  if (!currentPostData || selectedImages.length === 0) {
    showBanner("Please select image(s) first.", "warning");
    return;
  }
  const title = document.getElementById("finalTitleInput").value;
  const desc = document.getElementById("descriptionInput").value;
  const tags = (currentPostData.tags || []).join(", ");

  alert(`[Pixiv Post Preview]\n\nTitle:\n${title}\n\nTags:\n${tags}\n\nCaption/Description:\n${desc}\n\nTotal Images: ${selectedImages.length}`);
}

// Reset Form
function resetForm() {
  currentPostData = null;
  document.getElementById("modelBadge").textContent = "Not Detected";
  document.getElementById("metaCharacter").textContent = "--";
  document.getElementById("metaClothing").textContent = "--";
  document.getElementById("metaAppearance").textContent = "--";
  document.getElementById("metaPose").textContent = "--";
  document.getElementById("metaExpression").textContent = "--";
  document.getElementById("metaMaterials").textContent = "--";
  document.getElementById("titleCandidatesList").innerHTML = '<div class="text-xs text-gray-500 py-3 text-center">Title suggestions will appear once images are loaded</div>';
  document.getElementById("finalTitleInput").value = "";
  document.getElementById("descriptionInput").value = "";
  document.getElementById("tagsContainer").innerHTML = '<span class="text-xs text-gray-500 italic p-1">No tags available</span>';
  document.getElementById("tagCountBadge").textContent = "0 / 10";
}

// Notifications
function showBanner(msg, type = "info") {
  const banner = document.getElementById("statusBanner");
  const msgEl = document.getElementById("statusMessage");
  banner.classList.remove("hidden", "bg-blue-900/80", "bg-green-900/80", "bg-red-900/80", "bg-yellow-900/80", "text-blue-200", "text-green-200", "text-red-200", "text-yellow-200");

  let icon = '<i class="fa-solid fa-circle-info"></i>';
  if (type === "success") {
    banner.classList.add("bg-green-900/80", "text-green-200");
    icon = '<i class="fa-solid fa-circle-check"></i>';
  } else if (type === "error") {
    banner.classList.add("bg-red-900/80", "text-red-200");
    icon = '<i class="fa-solid fa-triangle-exclamation"></i>';
  } else if (type === "warning") {
    banner.classList.add("bg-yellow-900/80", "text-yellow-200");
    icon = '<i class="fa-solid fa-circle-exclamation"></i>';
  } else {
    banner.classList.add("bg-blue-900/80", "text-blue-200");
  }

  msgEl.innerHTML = `${icon} <span>${msg}</span>`;
}

function hideStatusBanner() {
  document.getElementById("statusBanner").classList.add("hidden");
}

// Modal logic: Post History
async function openHistoryModal() {
  const modal = document.getElementById("historyModal");
  modal.classList.remove("hidden");
  const list = document.getElementById("historyList");
  list.innerHTML = '<div class="text-center text-gray-400 py-8">Loading history...</div>';

  try {
    const res = await fetch("/api/history");
    const posts = await res.json();
    if (posts.length === 0) {
      list.innerHTML = '<div class="text-center text-gray-500 py-8">No posting history yet</div>';
      return;
    }

    list.innerHTML = "";
    posts.forEach(p => {
      const item = document.createElement("div");
      item.className = "p-3 rounded-lg bg-[#0d1117] border border-[#30363d] text-xs space-y-1";
      const isSuccess = p.status === "success";
      item.innerHTML = `
        <div class="flex items-center justify-between">
          <span class="font-bold text-white text-sm">${p.title}</span>
          <span class="px-2 py-0.5 rounded font-mono ${isSuccess ? 'bg-green-900/50 text-green-400 border border-green-700' : 'bg-red-900/50 text-red-400 border border-red-700'}">
            ${isSuccess ? 'Success' : 'Failed'}
          </span>
        </div>
        <div class="text-gray-400 flex items-center gap-3">
          <span><i class="fa-regular fa-clock"></i> ${p.created_at}</span>
          <span>Pixiv ID: ${p.pixiv_id || 'None'}</span>
        </div>
        <div class="text-gray-300">
          <span class="text-gray-500">Tags:</span> ${p.tags.join(", ")}
        </div>
        ${p.error_message ? `<div class="text-red-400 mt-1">Error: ${p.error_message}</div>` : ''}
      `;
      list.appendChild(item);
    });
  } catch (err) {
    list.innerHTML = `<div class="text-red-400 py-4">Failed to fetch history: ${err.message}</div>`;
  }
}

function closeHistoryModal() {
  document.getElementById("historyModal").classList.add("hidden");
}

// Toggle fields depending on selected auth mode
function togglePixivAuthFields() {
  const mode = document.getElementById("pixivAuthModeSelect").value;
  const tokenField = document.getElementById("pixivTokenField");
  const cookieField = document.getElementById("pixivCookieField");

  if (mode === "cookie") {
    if (cookieField) cookieField.classList.remove("hidden");
    if (tokenField) tokenField.classList.add("hidden");
  } else if (mode === "api") {
    if (tokenField) tokenField.classList.remove("hidden");
    if (cookieField) cookieField.classList.add("hidden");
  } else {
    if (cookieField) cookieField.classList.add("hidden");
    if (tokenField) tokenField.classList.add("hidden");
  }
}

// Modal logic: Settings, Pixiv Account & Tag Dictionary
async function openSettingsModal() {
  document.getElementById("settingsModal").classList.remove("hidden");
  try {
    const resData = await fetch("/api/get_data_files");
    const data = await resData.json();
    document.getElementById("tagsJsonEditor").value = JSON.stringify(data.tags, null, 2);
    document.getElementById("fixedTagsJsonEditor").value = JSON.stringify(data.fixed_tags, null, 2);
    document.getElementById("charactersJsonEditor").value = JSON.stringify(data.characters, null, 2);

    // Load account settings
    const resSettings = await fetch("/api/settings");
    const settings = await resSettings.json();
    const pixiv = settings.pixiv || {};

    document.getElementById("pixivAuthModeSelect").value = pixiv.auth_mode || "dry_run";
    document.getElementById("pixivUserIdInput").value = pixiv.user_id || "";
    if (document.getElementById("pixivCookieInput")) {
      document.getElementById("pixivCookieInput").value = pixiv.cookie || "";
    }
    if (document.getElementById("pixivRefreshTokenInput")) {
      document.getElementById("pixivRefreshTokenInput").value = pixiv.refresh_token || "";
    }
    togglePixivAuthFields();
  } catch (err) {
    alert("Failed to load settings: " + err.message);
  }
}

function closeSettingsModal() {
  document.getElementById("settingsModal").classList.add("hidden");
}

async function saveSettingsData() {
  try {
    const tags = JSON.parse(document.getElementById("tagsJsonEditor").value);
    const fixedTags = JSON.parse(document.getElementById("fixedTagsJsonEditor").value);
    const characters = JSON.parse(document.getElementById("charactersJsonEditor").value);

    // Save dictionary files
    await fetch("/api/save_data_files", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        tags: tags,
        fixed_tags: fixedTags,
        characters: characters
      })
    });

    // Save account & settings
    const authMode = document.getElementById("pixivAuthModeSelect").value;
    const userId = document.getElementById("pixivUserIdInput").value.trim();
    const cookieVal = document.getElementById("pixivCookieInput") ? document.getElementById("pixivCookieInput").value.trim() : "";
    const refreshToken = document.getElementById("pixivRefreshTokenInput") ? document.getElementById("pixivRefreshTokenInput").value.trim() : "";

    await fetch("/api/save_settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        pixiv: {
          auth_mode: authMode,
          user_id: userId,
          cookie: cookieVal,
          refresh_token: refreshToken,
          r18_default: true,
          ai_generated_default: true
        }
      })
    });

    showBanner("Saved account configuration and tag dictionaries.", "success");
    closeSettingsModal();
    loadSettingsConfig();
    if (currentPostData) {
      regenerateTags();
    }
  } catch (err) {
    alert("Save error (check JSON syntax): " + err.message);
  }
}

async function loadSettingsConfig() {
  try {
    const res = await fetch("/api/settings");
    const s = await res.json();
    const pixiv = s.pixiv || {};
    const mode = pixiv.auth_mode || "dry_run";
    const userId = pixiv.user_id ? ` (${pixiv.user_id})` : "";

    const label = document.getElementById("uploadModeLabel");
    const accountBadge = document.getElementById("pixivAccountLabel");

    if (mode === "dry_run") {
      label.textContent = "Dry Run (Safe Verification)";
      label.className = "py-1.5 px-2 bg-[#0d1117] border border-blue-800/60 rounded text-blue-400 font-medium";
      if (accountBadge) accountBadge.textContent = `Account: Dry Run${userId}`;
    } else if (mode === "cookie") {
      label.textContent = `Pixiv Session Cookie${userId}`;
      label.className = "py-1.5 px-2 bg-[#0d1117] border border-green-800/60 rounded text-green-400 font-medium";
      if (accountBadge) accountBadge.textContent = `Account: Cookie Login${userId}`;
    } else {
      label.textContent = `Pixiv API${userId}`;
      label.className = "py-1.5 px-2 bg-[#0d1117] border border-green-800/60 rounded text-green-400 font-medium";
      if (accountBadge) accountBadge.textContent = `Account: Live API${userId}`;
    }
  } catch (err) {}
}
