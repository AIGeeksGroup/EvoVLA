
import { OrbitControls } from "./OrbitControls.js";
import * as SPLAT from "https://cdn.jsdelivr.net/npm/gsplat@latest";
const bucket = "models"

function shuffleArray(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
}

async function listFolders() {
    // Interacting with the google cloud storage APIS requires full on node installatiosn and such so meh, just parse the returned XML :) 
    let xmlDoc = await fetch(bucket).then(response => response.text()).then(xmlString => {
        const parser = new DOMParser();
        const xmlDoc = parser.parseFromString(xmlString, 'text/xml');
        return xmlDoc;
    }).catch(error => {
        console.error('Error:', error);
    });
    const contentsElements = xmlDoc.getElementsByTagName('Contents');
    let nodeArray = [].slice.call(contentsElements);

    function xmlToKey(folder) {
        let key = folder.getElementsByTagName("Key")[0].textContent;
        let parts = key.split("/");
        return parts[0] + "/" + parts[1];
    }

    const keys = [...new Set(nodeArray.map(f => xmlToKey(f)))];
    const filteredKeys = keys.filter(f => !f.includes("."))
    return filteredKeys
}

async function listLocalFolders() { // NEW
    const keys = [
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
        "13",
        "14",
    ]
    return keys
}

export function createSplatView(splatParent) {
    const parentDiv = document.getElementById(splatParent);
    const canvas = parentDiv.querySelector("canvas");
    const progressDialog = parentDiv.querySelector("#progress-dialog");
    const progressIndicator = progressDialog.querySelector("#progress-indicator");
    const view = new Object();
    view.canvas = canvas;
    view.progressDialog = progressDialog;
    view.progressIndicator = progressIndicator;
    view.runningAnimation = null;
    view.loading = false;
    view.lastClick = new Date()
    return view;
}

export async function setSplatScene(name, view) {
    view.loading = true;
    view.lastClick = new Date();

    const startRadius = 1

    const cameraData = new SPLAT.CameraData();
    cameraData.fx = 0.6 * startRadius * view.canvas.offsetWidth
    cameraData.fy = 0.6 * startRadius * view.canvas.offsetHeight

    const camera = new SPLAT.Camera(cameraData);
    const renderer = new SPLAT.WebGLRenderer(view.canvas);
    const scene = new SPLAT.Scene();

    // Load and convert ply from url
    view.progressDialog.show();
    view.progressIndicator.value = 0.0;

    const splat = await SPLAT.PLYLoader.LoadAsync(bucket + "/" + name, scene, (progress) => (view.progressIndicator.value = progress * 100));

    // NerfStudio and gsplat.js don't agree on coordinate frames, just rotate after loading for now.
    // Could try to add this as a preprocessing step in the future, this is not ideal :)
    // Also compensate for the slight y offset the nerf is traine with - which means a small rotation 
    // on the x-axis.
    // const rotation = new SPLAT.Vector3(Math.PI - Math.PI / 20.0, Math.PI, 0);
    // const rotation = new SPLAT.Vector3(Math.PI - Math.PI / 20.0, 0, 0);  // NEW!
    const rotation = new SPLAT.Vector3(Math.PI/2, 1.5*Math.PI, 0); // Rotate around Y and then X
    splat.rotation = SPLAT.Quaternion.FromEuler(rotation);
    splat.applyRotation();

    // renderer.setClearColor(1.0, 1.0, 1.0, 1.0); // NEW! set clear color to white
    renderer.backgroundColor = new SPLAT.Color32(255, 255, 255, 255);  // NEW

    view.progressDialog.close();
    // Not sure if these are meant to be public, but it's javascript anything goes :)
    var controls = new OrbitControls(camera, view.canvas, /*alpha=*/0.0, /*beta=*/0.0, /*radius=*/startRadius, /*enableKeyboardControls=*/false);
    controls.minAngle = -45
    controls.maxAngle = 45
    controls.minZoom = 0.8
    controls.maxZoom = 1.6
    controls.zoomSpeed = 0.03
    controls.panSpeed = 0.2
    controls.orbitSpeed = 1.75
    controls.maxPanDistance = 0.05

    const newAnimation = view.runningAnimation == null;

    if (newAnimation) {
        view.canvas.addEventListener("mousedown", function () {
            view.lastClick = new Date();
            view.interacting = true;
        });
        view.canvas.addEventListener("mouseup", function () {
            view.lastClick = new Date();
            view.interacting = false;
        });
    }

    // Render loop
    let previousTimestamp = undefined;
    let previousDeltaTime = undefined;
    view.runningAnimation = (timestamp) => {
        var deltaTime = 0.0;
        if (previousTimestamp !== undefined) {
            deltaTime = (timestamp - previousTimestamp) / 1000;
        }
        // After alt-tabbing, deltaTime can go up very high... just resume the animation where it was.
        if (deltaTime > 0.1 && previousDeltaTime !== undefined) {
            deltaTime = previousDeltaTime;
        }
        previousTimestamp = timestamp;
        previousDeltaTime = deltaTime;

        if (!view.interacting) {
            const timeToSpin = 0.5;
            const accelTime = 4.0;
            const maxSpinSpeed = 0.2;

            const timeSinceClick = view.lastClick == undefined ? undefined : (new Date() - view.lastClick) / 1000.0;
            if (timeSinceClick > timeToSpin || timeSinceClick === undefined) {
                const speed = timeSinceClick === undefined ? maxSpinSpeed : Math.min(Math.max(timeSinceClick / accelTime - timeToSpin, 0.0), 1.0) * maxSpinSpeed;
                controls.rotateCameraAngle(speed * deltaTime, 0.0);
            }
        }

        controls.update();

        renderer.render(scene, camera);
        requestAnimationFrame(view.runningAnimation);
    };

    if (newAnimation) {
        requestAnimationFrame(view.runningAnimation);
    }

    view.loading = false;
}

export async function setupCarousel(view, carousel) {
    // let files = await listFolders();
    let files = await listLocalFolders();
    console.log(files)

    // shuffleArray(files);

    const prototype = carousel.querySelector("#splat-carousel-prototype");
    const elements = Object.fromEntries(
        files.map(f => [f, prototype.firstElementChild.cloneNode(true)])
    );

    async function onClickSplatThumb(splatName) {
        if (view.loading) {
            // Only allow one splat at a time to load otherwise things get messy.
            return;
        }
        const elem = elements[splatName];
        if (elem.classList.contains("active")) {
            return;
        }

        const itemsParent = carousel.getElementsByClassName("splat-carousel-items")[0];
        const items = [...itemsParent.getElementsByClassName('splat-carousel-item')];
        currentIndex = items.indexOf(elem);

        elem.classList.add("loading");

        await setSplatScene(splatName + "/splat.ply", view)

        Object.values(elements).forEach(e => {
            e.classList.remove("active");
        });
        elem.classList.remove("loading");
        elem.classList.add("active");
    }

    // TODO: probably want to download a bunch of metadata-y thing here.
    for (var i = 0; i < files.length; ++i) {
        const file = files[i];

        // Wrap in function to capture CURRENT file.
        // Js is bad :)
        function setup(file) {
            const card = elements[file];
            var startScroll = undefined;

            card.addEventListener("mousedown", function () { startScroll = itemsParent.scrollLeft; });
            card.addEventListener("mouseup", function () {
                // console.log(Math.abs(itemsParent.scrollLeft - startScroll));
                if (Math.abs(itemsParent.scrollLeft - startScroll) < 10) {
                    onClickSplatThumb(file);
                }
            });

            const img = card.querySelector("img");
            img.src = bucket + "/" + file + "/input.png";
            
            let isAnimating = false;
            let latestEvent = null;

            card.addEventListener('pointermove', (e) => {
                latestEvent = e;
                if (!isAnimating) {
                    isAnimating = true;
                    requestAnimationFrame(updateCardTransform);
                }
            });

            function updateCardTransform() {
                const e = latestEvent;
                if (e === null || e === undefined) {
                    isAnimating = false;
                    return;
                }
                const cardRect = card.getBoundingClientRect();
                const centerX = cardRect.left + cardRect.width / 2;
                const centerY = cardRect.top + cardRect.height / 2;

                const mouseX = e.clientX - centerX;
                const mouseY = e.clientY - centerY;

                const rotateY = (mouseX / cardRect.width) * 35;
                const rotateX = -(mouseY / cardRect.height) * 35;

                card.style.transform = `translateZ(15px) rotateY(${rotateY}deg) rotateX(${rotateX}deg)`;

                isAnimating = false;
                if (latestEvent !== e) { // Check for new event
                    requestAnimationFrame(updateCardTransform);
                }
            }

            card.addEventListener('mouseleave', () => {
                latestEvent = null; // Clear event to stop animation
                card.style.transform = ''; // Reset on leave
            });
            prototype.parentNode.appendChild(card);
        }

        setup(file);
    }

    prototype.remove();

    const itemsParent = carousel.getElementsByClassName("splat-carousel-items")[0];
    const items = [...itemsParent.getElementsByClassName('splat-carousel-item')];
    let currentIndex = 0;

    function scrollToTarget() {
        items[currentIndex].scrollIntoView({ behavior: "smooth", inline: "center", block: "nearest" });
    }

    carousel.querySelector('.splat-carousel-button.left').addEventListener('mousedown', () => {
        currentIndex = (currentIndex + items.length - 2) % items.length;
        scrollToTarget();
    });

    carousel.querySelector('.splat-carousel-button.right').addEventListener('mousedown', () => {
        currentIndex = (currentIndex + 2) % items.length;
        scrollToTarget();
    });

    let mouseDown = false;
    let startX, scrollLeft;

    const startDragging = (e) => {
        mouseDown = true;
        startX = e.pageX - itemsParent.offsetLeft;
        scrollLeft = itemsParent.scrollLeft;
    }

    const stopDragging = (e) => {
        e.preventDefault();
        mouseDown = false;
    }

    const move = (e) => {
        e.preventDefault();
        if (!mouseDown) { return; }
        const x = e.pageX - itemsParent.offsetLeft;
        const scroll = x - startX;
        itemsParent.scrollLeft = scrollLeft - scroll;
    }

    // Add the event listeners
    itemsParent.addEventListener('mousemove', move, false);
    itemsParent.addEventListener('mousedown', startDragging, false);
    itemsParent.addEventListener('mouseup', stopDragging, false);
    itemsParent.addEventListener('mouseleave', stopDragging, false);

    // Activate the first thumbnail.
    onClickSplatThumb(files[0]);
}
