// Enhanced Three.js effects for different sections
class EnhancedEffects {
    constructor() {
        this.scenes = {};
        this.renderers = {};
        this.cameras = {};
        this.init();
    }

    init() {
        this.createHeroEffect();
        this.createAboutEffect();
        this.createFeaturesEffect();
        this.createDemoEffect();
        this.createTeamEffect();
        this.addCursorEffect();
        this.addElementAnimations();
        this.addStickyButtons();
    }

    createHeroEffect() {
        const canvas = document.getElementById('three-canvas');
        if (!canvas) return;

        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({ canvas, alpha: true });
        renderer.setSize(window.innerWidth, window.innerHeight);

        // Create floating geometric shapes
        const geometry = new THREE.IcosahedronGeometry(1, 0);
        const material = new THREE.MeshBasicMaterial({ 
            color: 0x667eea, 
            wireframe: true,
            transparent: true,
            opacity: 0.3
        });

        const shapes = [];
        for (let i = 0; i < 20; i++) {
            const mesh = new THREE.Mesh(geometry, material);
            mesh.position.set(
                (Math.random() - 0.5) * 50,
                (Math.random() - 0.5) * 50,
                (Math.random() - 0.5) * 50
            );
            mesh.rotation.set(Math.random() * Math.PI, Math.random() * Math.PI, 0);
            scene.add(mesh);
            shapes.push(mesh);
        }

        camera.position.z = 30;

        const animate = () => {
            requestAnimationFrame(animate);
            shapes.forEach((shape, i) => {
                shape.rotation.x += 0.01 + i * 0.001;
                shape.rotation.y += 0.01 + i * 0.001;
                shape.position.y += Math.sin(Date.now() * 0.001 + i) * 0.01;
            });
            renderer.render(scene, camera);
        };
        animate();

        this.scenes.hero = { scene, camera, renderer, shapes };
    }

    createAboutEffect() {
        const aboutSection = document.getElementById('about');
        if (!aboutSection) return;

        const canvas = document.createElement('canvas');
        canvas.style.position = 'absolute';
        canvas.style.top = '0';
        canvas.style.left = '0';
        canvas.style.width = '100%';
        canvas.style.height = '100%';
        canvas.style.pointerEvents = 'none';
        canvas.style.zIndex = '1';
        aboutSection.style.position = 'relative';
        aboutSection.appendChild(canvas);

        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({ canvas, alpha: true });
        renderer.setSize(aboutSection.offsetWidth, aboutSection.offsetHeight);

        // Create neural network visualization
        const nodes = [];
        const connections = [];
        
        for (let i = 0; i < 15; i++) {
            const nodeGeometry = new THREE.SphereGeometry(0.1, 8, 8);
            const nodeMaterial = new THREE.MeshBasicMaterial({ color: 0x4ecdc4 });
            const node = new THREE.Mesh(nodeGeometry, nodeMaterial);
            node.position.set(
                (Math.random() - 0.5) * 20,
                (Math.random() - 0.5) * 10,
                (Math.random() - 0.5) * 5
            );
            scene.add(node);
            nodes.push(node);
        }

        camera.position.z = 15;

        const animate = () => {
            requestAnimationFrame(animate);
            nodes.forEach(node => {
                node.position.x += Math.sin(Date.now() * 0.001) * 0.01;
                node.position.y += Math.cos(Date.now() * 0.001) * 0.01;
            });
            renderer.render(scene, camera);
        };
        animate();
    }

    createFeaturesEffect() {
        const featuresSection = document.getElementById('features');
        if (!featuresSection) return;

        const canvas = document.createElement('canvas');
        canvas.style.position = 'absolute';
        canvas.style.top = '0';
        canvas.style.left = '0';
        canvas.style.width = '100%';
        canvas.style.height = '100%';
        canvas.style.pointerEvents = 'none';
        canvas.style.zIndex = '1';
        canvas.style.opacity = '0.1';
        featuresSection.style.position = 'relative';
        featuresSection.appendChild(canvas);

        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({ canvas, alpha: true });
        renderer.setSize(featuresSection.offsetWidth, featuresSection.offsetHeight);

        // Create matrix rain effect
        const geometry = new THREE.PlaneGeometry(0.5, 0.5);
        const material = new THREE.MeshBasicMaterial({ 
            color: 0x00ff00,
            transparent: true,
            opacity: 0.7
        });

        const drops = [];
        for (let i = 0; i < 50; i++) {
            const drop = new THREE.Mesh(geometry, material);
            drop.position.set(
                (Math.random() - 0.5) * 30,
                Math.random() * 20 + 10,
                (Math.random() - 0.5) * 10
            );
            scene.add(drop);
            drops.push(drop);
        }

        camera.position.z = 20;

        const animate = () => {
            requestAnimationFrame(animate);
            drops.forEach(drop => {
                drop.position.y -= 0.1;
                if (drop.position.y < -10) {
                    drop.position.y = 20;
                    drop.position.x = (Math.random() - 0.5) * 30;
                }
            });
            renderer.render(scene, camera);
        };
        animate();
    }

    createDemoEffect() {
        const canvas = document.getElementById('demo-canvas');
        if (!canvas) return;

        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({ canvas, alpha: true });
        renderer.setSize(window.innerWidth, window.innerHeight);

        // Create terminal-like effect
        const lines = [];
        for (let i = 0; i < 50; i++) {
            const geometry = new THREE.PlaneGeometry(20, 0.1);
            const material = new THREE.MeshBasicMaterial({ 
                color: 0x00ff00,
                transparent: true,
                opacity: 0.3
            });
            const line = new THREE.Mesh(geometry, material);
            line.position.set(
                (Math.random() - 0.5) * 40,
                (Math.random() - 0.5) * 30,
                (Math.random() - 0.5) * 20
            );
            scene.add(line);
            lines.push(line);
        }

        camera.position.z = 30;

        const animate = () => {
            requestAnimationFrame(animate);
            lines.forEach(line => {
                line.position.x += 0.02;
                if (line.position.x > 25) line.position.x = -25;
                line.material.opacity = 0.1 + Math.sin(Date.now() * 0.001) * 0.2;
            });
            renderer.render(scene, camera);
        };
        animate();
    }

    createTeamEffect() {
        const canvas = document.getElementById('team-canvas');
        if (!canvas) return;

        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({ canvas, alpha: true });
        renderer.setSize(window.innerWidth, window.innerHeight);

        // Create network connections
        const nodes = [];
        for (let i = 0; i < 20; i++) {
            const nodeGeometry = new THREE.SphereGeometry(0.1, 8, 8);
            const nodeMaterial = new THREE.MeshBasicMaterial({ 
                color: 0x4ecdc4,
                transparent: true,
                opacity: 0.6
            });
            const node = new THREE.Mesh(nodeGeometry, nodeMaterial);
            node.position.set(
                (Math.random() - 0.5) * 40,
                (Math.random() - 0.5) * 25,
                (Math.random() - 0.5) * 15
            );
            scene.add(node);
            nodes.push(node);
        }

        // Create connections
        const connections = [];
        for (let i = 0; i < nodes.length; i++) {
            for (let j = i + 1; j < nodes.length; j++) {
                if (Math.random() > 0.7) {
                    const geometry = new THREE.BufferGeometry();
                    const positions = new Float32Array([
                        nodes[i].position.x, nodes[i].position.y, nodes[i].position.z,
                        nodes[j].position.x, nodes[j].position.y, nodes[j].position.z
                    ]);
                    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
                    const material = new THREE.LineBasicMaterial({ 
                        color: 0x667eea,
                        transparent: true,
                        opacity: 0.2
                    });
                    const line = new THREE.Line(geometry, material);
                    scene.add(line);
                    connections.push(line);
                }
            }
        }

        camera.position.z = 30;

        const animate = () => {
            requestAnimationFrame(animate);
            nodes.forEach(node => {
                node.position.x += Math.sin(Date.now() * 0.001) * 0.01;
                node.position.y += Math.cos(Date.now() * 0.001) * 0.01;
            });
            renderer.render(scene, camera);
        };
        animate();
    }

    addCursorEffect() {
        // Create custom heart cursor
        const style = document.createElement('style');
        style.textContent = `
            * {
                cursor: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="red"><path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z"/></svg>') 12 12, auto !important;
            }
            
            .btn, .nav-menu a, .feature-card, .team-member {
                cursor: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="gold"><path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z"/></svg>') 12 12, pointer !important;
            }
        `;
        document.head.appendChild(style);

        // Add cursor trail effect
        const trail = [];
        const trailLength = 10;

        document.addEventListener('mousemove', (e) => {
            trail.push({ x: e.clientX, y: e.clientY, time: Date.now() });
            if (trail.length > trailLength) trail.shift();

            // Remove old trail elements
            document.querySelectorAll('.cursor-trail').forEach(el => el.remove());

            // Create new trail
            trail.forEach((point, index) => {
                const trailElement = document.createElement('div');
                trailElement.className = 'cursor-trail';
                trailElement.style.cssText = `
                    position: fixed;
                    left: ${point.x}px;
                    top: ${point.y}px;
                    width: ${10 - index}px;
                    height: ${10 - index}px;
                    background: radial-gradient(circle, rgba(255,0,0,${0.8 - index * 0.08}) 0%, transparent 70%);
                    border-radius: 50%;
                    pointer-events: none;
                    z-index: 9999;
                    transform: translate(-50%, -50%);
                `;
                document.body.appendChild(trailElement);

                setTimeout(() => trailElement.remove(), 500);
            });
        });
    }

    addElementAnimations() {
        // Add floating animation to all cards
        const cards = document.querySelectorAll('.feature-card, .team-member, .step');
        cards.forEach((card, index) => {
            card.style.animation = `float ${3 + index * 0.5}s ease-in-out infinite`;
            card.style.animationDelay = `${index * 0.2}s`;
        });

        // Add pulse animation to stats
        const stats = document.querySelectorAll('.stat-number');
        stats.forEach((stat, index) => {
            stat.style.animation = `pulse ${2 + index * 0.3}s ease-in-out infinite`;
        });

        // Add glow effect to icons
        const icons = document.querySelectorAll('.feature-icon, .member-avatar');
        icons.forEach(icon => {
            icon.style.animation = 'glow 2s ease-in-out infinite alternate';
        });
    }

    addStickyButtons() {
        const heroButtons = document.querySelector('.hero-buttons');
        const hero = document.querySelector('.hero');
        
        if (!heroButtons || !hero) return;

        window.addEventListener('scroll', () => {
            const heroRect = hero.getBoundingClientRect();
            const scrolled = window.pageYOffset;
            
            if (heroRect.bottom < 0) {
                // Hero section is out of view
                heroButtons.style.position = 'fixed';
                heroButtons.style.top = '20px';
                heroButtons.style.right = '20px';
                heroButtons.style.transform = 'scale(0.8)';
                heroButtons.style.flexDirection = 'column';
                heroButtons.style.gap = '0.5rem';
                heroButtons.style.zIndex = '1000';
            } else {
                // Hero section is in view
                heroButtons.style.position = 'sticky';
                heroButtons.style.top = '50vh';
                heroButtons.style.right = 'auto';
                heroButtons.style.transform = 'scale(1)';
                heroButtons.style.flexDirection = 'row';
                heroButtons.style.gap = '1rem';
            }
        });
    }
}

// CSS animations
const animationStyles = document.createElement('style');
animationStyles.textContent = `
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    @keyframes glow {
        0% { box-shadow: 0 0 20px rgba(102, 126, 234, 0.5); }
        100% { box-shadow: 0 0 30px rgba(102, 126, 234, 0.8), 0 0 40px rgba(102, 126, 234, 0.3); }
    }
    
    .feature-card:hover {
        transform: translateY(-15px) scale(1.02) !important;
        box-shadow: 0 20px 40px rgba(0,0,0,0.2) !important;
        transition: all 0.3s ease !important;
    }
    
    .team-member:hover {
        transform: translateY(-10px) rotate(2deg) !important;
        transition: all 0.3s ease !important;
    }
    
    .btn:hover {
        transform: translateY(-3px) scale(1.05) !important;
        box-shadow: 0 10px 25px rgba(0,0,0,0.2) !important;
    }
`;
document.head.appendChild(animationStyles);

// Initialize enhanced effects when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new EnhancedEffects();
});