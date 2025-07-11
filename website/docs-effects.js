// Documentation page Three.js effects
class DocsEffects {
    constructor() {
        this.init();
    }

    init() {
        this.createDocsBackground();
        this.addScrollEffects();
        this.addSidebarInteraction();
    }

    createDocsBackground() {
        const canvas = document.getElementById('docs-canvas');
        if (!canvas) return;

        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({ canvas, alpha: true });
        renderer.setSize(window.innerWidth - 280, window.innerHeight - 80);

        // Create flowing data streams
        const streams = [];
        for (let i = 0; i < 30; i++) {
            const geometry = new THREE.CylinderGeometry(0.02, 0.02, 2, 8);
            const material = new THREE.MeshBasicMaterial({ 
                color: new THREE.Color().setHSL(0.6 + Math.random() * 0.2, 0.7, 0.5),
                transparent: true,
                opacity: 0.6
            });
            const stream = new THREE.Mesh(geometry, material);
            
            stream.position.set(
                (Math.random() - 0.5) * 50,
                (Math.random() - 0.5) * 30,
                (Math.random() - 0.5) * 20
            );
            stream.rotation.z = Math.random() * Math.PI;
            
            scene.add(stream);
            streams.push({
                mesh: stream,
                speed: 0.01 + Math.random() * 0.02,
                direction: Math.random() > 0.5 ? 1 : -1
            });
        }

        // Add floating code symbols
        const symbols = ['{ }', '< >', '[ ]', '( )', '=', '+', '-', '*'];
        const codeElements = [];
        
        symbols.forEach((symbol, index) => {
            const canvas2d = document.createElement('canvas');
            const context = canvas2d.getContext('2d');
            canvas2d.width = 64;
            canvas2d.height = 64;
            
            context.fillStyle = '#4ecdc4';
            context.font = '24px monospace';
            context.textAlign = 'center';
            context.fillText(symbol, 32, 40);
            
            const texture = new THREE.CanvasTexture(canvas2d);
            const material = new THREE.SpriteMaterial({ 
                map: texture,
                transparent: true,
                opacity: 0.3
            });
            const sprite = new THREE.Sprite(material);
            
            sprite.position.set(
                (Math.random() - 0.5) * 40,
                (Math.random() - 0.5) * 25,
                (Math.random() - 0.5) * 15
            );
            sprite.scale.set(2, 2, 1);
            
            scene.add(sprite);
            codeElements.push({
                sprite: sprite,
                rotationSpeed: 0.005 + Math.random() * 0.01,
                floatSpeed: 0.01 + Math.random() * 0.02
            });
        });

        camera.position.z = 25;

        const animate = () => {
            requestAnimationFrame(animate);
            
            // Animate streams
            streams.forEach(stream => {
                stream.mesh.position.y += stream.speed * stream.direction;
                stream.mesh.rotation.x += 0.01;
                
                if (stream.mesh.position.y > 20) {
                    stream.mesh.position.y = -20;
                } else if (stream.mesh.position.y < -20) {
                    stream.mesh.position.y = 20;
                }
            });
            
            // Animate code elements
            codeElements.forEach(element => {
                element.sprite.material.rotation += element.rotationSpeed;
                element.sprite.position.y += Math.sin(Date.now() * element.floatSpeed) * 0.01;
            });
            
            renderer.render(scene, camera);
        };
        animate();

        // Handle window resize
        window.addEventListener('resize', () => {
            camera.aspect = (window.innerWidth - 280) / (window.innerHeight - 80);
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth - 280, window.innerHeight - 80);
        });
    }

    addScrollEffects() {
        // Smooth scroll for navigation links
        document.querySelectorAll('.docs-nav a').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const targetId = link.getAttribute('href');
                const targetElement = document.querySelector(targetId);
                
                if (targetElement) {
                    targetElement.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                    
                    // Update active link
                    document.querySelectorAll('.docs-nav a').forEach(a => a.classList.remove('active'));
                    link.classList.add('active');
                }
            });
        });

        // Intersection observer for section highlighting
        const sections = document.querySelectorAll('.docs-section');
        const navLinks = document.querySelectorAll('.docs-nav a');
        
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const id = entry.target.getAttribute('id');
                    navLinks.forEach(link => {
                        link.classList.remove('active');
                        if (link.getAttribute('href') === `#${id}`) {
                            link.classList.add('active');
                        }
                    });
                }
            });
        }, { threshold: 0.3 });

        sections.forEach(section => observer.observe(section));

        // Parallax effect for content cards
        window.addEventListener('scroll', () => {
            const scrolled = window.pageYOffset;
            const cards = document.querySelectorAll('.content-card, .result-card, .step-card');
            
            cards.forEach((card, index) => {
                const speed = 0.1 + (index % 3) * 0.05;
                card.style.transform = `translateY(${scrolled * speed}px)`;
            });
        });
    }

    addSidebarInteraction() {
        // Add hover effects to sidebar links
        const sidebarLinks = document.querySelectorAll('.docs-nav a');
        
        sidebarLinks.forEach(link => {
            link.addEventListener('mouseenter', () => {
                link.style.transform = 'translateX(10px) scale(1.05)';
                link.style.boxShadow = '0 5px 15px rgba(78, 205, 196, 0.3)';
            });
            
            link.addEventListener('mouseleave', () => {
                if (!link.classList.contains('active')) {
                    link.style.transform = 'translateX(0) scale(1)';
                    link.style.boxShadow = 'none';
                }
            });
        });

        // Add progress indicator
        const progressBar = document.createElement('div');
        progressBar.style.cssText = `
            position: fixed;
            top: 80px;
            left: 0;
            width: 4px;
            height: 0%;
            background: linear-gradient(to bottom, #4ecdc4, #667eea);
            z-index: 1001;
            transition: height 0.3s ease;
        `;
        document.body.appendChild(progressBar);

        window.addEventListener('scroll', () => {
            const scrollTop = window.pageYOffset;
            const docHeight = document.documentElement.scrollHeight - window.innerHeight;
            const scrollPercent = (scrollTop / docHeight) * 100;
            progressBar.style.height = `${Math.min(scrollPercent, 100)}%`;
        });
    }
}

// Initialize documentation effects
document.addEventListener('DOMContentLoaded', () => {
    new DocsEffects();
    
    // Add typing effect to code blocks
    const codeBlocks = document.querySelectorAll('.code-block code');
    codeBlocks.forEach((block, index) => {
        const text = block.textContent;
        block.textContent = '';
        
        setTimeout(() => {
            let i = 0;
            const typeInterval = setInterval(() => {
                block.textContent += text.charAt(i);
                i++;
                if (i >= text.length) {
                    clearInterval(typeInterval);
                }
            }, 20);
        }, index * 1000);
    });
    
    // Add counter animation to stats
    const statValues = document.querySelectorAll('.stat-value, .metric-value');
    const animateCounter = (element, target, suffix = '') => {
        const isPercentage = target.includes('%');
        const numericTarget = parseFloat(target);
        let current = 0;
        const increment = numericTarget / 50;
        
        const timer = setInterval(() => {
            current += increment;
            if (current >= numericTarget) {
                current = numericTarget;
                clearInterval(timer);
            }
            element.textContent = current.toFixed(isPercentage ? 2 : 1) + (isPercentage ? '%' : suffix);
        }, 50);
    };
    
    // Intersection observer for stat animations
    const statsObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const target = entry.target.textContent;
                animateCounter(entry.target, target);
                statsObserver.unobserve(entry.target);
            }
        });
    });
    
    statValues.forEach(stat => {
        statsObserver.observe(stat);
    });
});