
/**
 * Behavioral Biometric Authentication System - Frontend
 * Captures behavioral patterns during user interactions
 */

class BiometricCapture {
    constructor() {
        this.patterns = {
            keystroke: [],
            mouse: [],
            touch: [],
            device: {}
        };
        this.isCapturing = false;
        this.startTime = null;
        this.init();
    }

    init() {
        this.captureDeviceFingerprint();
        this.setupEventListeners();
    }

    // Device Fingerprinting
    captureDeviceFingerprint() {
        try {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            ctx.textBaseline = 'top';
            ctx.font = '14px Arial';
            ctx.fillText('Device fingerprint', 2, 2);

            this.patterns.device = {
                userAgent: navigator.userAgent || 'unknown',
                language: navigator.language || 'unknown',
                platform: navigator.platform || 'unknown',
                screenResolution: `${screen.width}x${screen.height}`,
                colorDepth: screen.colorDepth || 24,
                timezone: Intl.DateTimeFormat().resolvedOptions().timeZone || 'unknown',
                canvasFingerprint: canvas.toDataURL(),
                hardwareConcurrency: navigator.hardwareConcurrency || 4,
                deviceMemory: navigator.deviceMemory || 8,
                cookieEnabled: navigator.cookieEnabled || false,
                doNotTrack: navigator.doNotTrack || 'unknown',
                plugins: Array.from(navigator.plugins || []).map(p => p.name),
                timestamp: Date.now()
            };

            console.log('Device fingerprint captured:', this.patterns.device);
            
            // Get IP and connection info (would need backend call)
            this.getNetworkInfo();
        } catch (error) {
            console.error('Error capturing device fingerprint:', error);
            // Fallback minimal device fingerprint
            this.patterns.device = {
                userAgent: navigator.userAgent || 'unknown',
                platform: navigator.platform || 'unknown',
                screenResolution: `${screen.width}x${screen.height}`,
                colorDepth: screen.colorDepth || 24,
                timestamp: Date.now()
            };
        }
    }

    async getNetworkInfo() {
        try {
            // This would need a backend endpoint to get real IP
            const response = await fetch('/api/network-info');
            const networkData = await response.json();
            if (networkData.success) {
                this.patterns.device.ipAddress = networkData.network_info.ip_address;
                this.patterns.device.userAgent = networkData.network_info.user_agent;
            }
        } catch (error) {
            console.log('Network info not available');
        }
    }

    setupEventListeners() {
        // Keystroke dynamics
        document.addEventListener('keydown', (e) => this.captureKeystroke(e, 'down'));
        document.addEventListener('keyup', (e) => this.captureKeystroke(e, 'up'));

        // Mouse dynamics
        document.addEventListener('mousedown', (e) => this.captureMouse(e, 'down'));
        document.addEventListener('mouseup', (e) => this.captureMouse(e, 'up'));
        document.addEventListener('mousemove', (e) => this.captureMouseMove(e));
        document.addEventListener('wheel', (e) => this.captureScroll(e));

        // Touch dynamics (mobile)
        document.addEventListener('touchstart', (e) => this.captureTouch(e, 'start'));
        document.addEventListener('touchend', (e) => this.captureTouch(e, 'end'));
        document.addEventListener('touchmove', (e) => this.captureTouchMove(e));

        // Device orientation (mobile)
        if (window.DeviceOrientationEvent) {
            window.addEventListener('deviceorientation', (e) => this.captureOrientation(e));
        }
    }

    startCapture() {
        this.isCapturing = true;
        this.startTime = Date.now();
        
        // Ensure device fingerprint is captured before starting
        this.captureDeviceFingerprint();
        
        this.patterns = {
            keystroke: [],
            mouse: [],
            touch: [],
            device: this.patterns.device
        };
    }

    stopCapture() {
        this.isCapturing = false;
        return this.getProcessedPatterns();
    }

    captureKeystroke(event, type) {
        if (!this.isCapturing) return;

        const keystrokeData = {
            key: event.key,
            code: event.code,
            type: type,
            timestamp: Date.now() - this.startTime,
            shiftKey: event.shiftKey,
            ctrlKey: event.ctrlKey,
            altKey: event.altKey,
            metaKey: event.metaKey
        };

        this.patterns.keystroke.push(keystrokeData);
    }

    captureMouse(event, type) {
        if (!this.isCapturing) return;

        const mouseData = {
            type: type,
            x: event.clientX,
            y: event.clientY,
            button: event.button,
            timestamp: Date.now() - this.startTime
        };

        this.patterns.mouse.push(mouseData);
    }

    captureMouseMove(event) {
        if (!this.isCapturing) return;

        const mouseMoveData = {
            type: 'move',
            x: event.clientX,
            y: event.clientY,
            movementX: event.movementX,
            movementY: event.movementY,
            timestamp: Date.now() - this.startTime
        };

        this.patterns.mouse.push(mouseMoveData);
    }

    captureScroll(event) {
        if (!this.isCapturing) return;

        const scrollData = {
            type: 'scroll',
            deltaX: event.deltaX,
            deltaY: event.deltaY,
            deltaZ: event.deltaZ,
            deltaMode: event.deltaMode,
            timestamp: Date.now() - this.startTime
        };

        this.patterns.mouse.push(scrollData);
    }

    captureTouch(event, type) {
        if (!this.isCapturing) return;

        for (let touch of event.touches) {
            const touchData = {
                type: type,
                x: touch.clientX,
                y: touch.clientY,
                radiusX: touch.radiusX || 0,
                radiusY: touch.radiusY || 0,
                rotationAngle: touch.rotationAngle || 0,
                force: touch.force || 0,
                timestamp: Date.now() - this.startTime,
                identifier: touch.identifier
            };

            this.patterns.touch.push(touchData);
        }
    }

    captureTouchMove(event) {
        if (!this.isCapturing) return;

        for (let touch of event.touches) {
            const touchMoveData = {
                type: 'move',
                x: touch.clientX,
                y: touch.clientY,
                radiusX: touch.radiusX || 0,
                radiusY: touch.radiusY || 0,
                rotationAngle: touch.rotationAngle || 0,
                force: touch.force || 0,
                timestamp: Date.now() - this.startTime,
                identifier: touch.identifier
            };

            this.patterns.touch.push(touchMoveData);
        }
    }

    captureOrientation(event) {
        if (!this.isCapturing) return;

        const orientationData = {
            alpha: event.alpha,
            beta: event.beta,
            gamma: event.gamma,
            timestamp: Date.now() - this.startTime
        };

        if (!this.patterns.orientation) {
            this.patterns.orientation = [];
        }
        this.patterns.orientation.push(orientationData);
    }

    getProcessedPatterns() {
        return {
            keystroke: this.processKeystrokePatterns(),
            mouse: this.processMousePatterns(),
            touch: this.processTouchPatterns(),
            device: this.patterns.device,
            orientation: this.patterns.orientation || []
        };
    }

    processKeystrokePatterns() {
        const keystrokes = this.patterns.keystroke;
        const features = {};

        // Calculate dwell times (key hold duration)
        const dwellTimes = {};
        const flightTimes = {};

        let lastKeyUp = null;

        for (let i = 0; i < keystrokes.length; i++) {
            const current = keystrokes[i];

            if (current.type === 'down') {
                // Find corresponding keyup
                for (let j = i + 1; j < keystrokes.length; j++) {
                    if (keystrokes[j].key === current.key && keystrokes[j].type === 'up') {
                        const dwellTime = keystrokes[j].timestamp - current.timestamp;
                        if (!dwellTimes[current.key]) dwellTimes[current.key] = [];
                        dwellTimes[current.key].push(dwellTime);
                        break;
                    }
                }

                // Calculate flight time from last key release
                if (lastKeyUp) {
                    const flightTime = current.timestamp - lastKeyUp.timestamp;
                    const keyPair = `${lastKeyUp.key}_${current.key}`;
                    if (!flightTimes[keyPair]) flightTimes[keyPair] = [];
                    flightTimes[keyPair].push(flightTime);
                }
            } else if (current.type === 'up') {
                lastKeyUp = current;
            }
        }

        // Calculate average dwell and flight times
        features.avgDwellTimes = {};
        for (let key in dwellTimes) {
            features.avgDwellTimes[key] = dwellTimes[key].reduce((a, b) => a + b, 0) / dwellTimes[key].length;
        }

        features.avgFlightTimes = {};
        for (let keyPair in flightTimes) {
            features.avgFlightTimes[keyPair] = flightTimes[keyPair].reduce((a, b) => a + b, 0) / flightTimes[keyPair].length;
        }

        features.typingSpeed = keystrokes.filter(k => k.type === 'down').length / (Date.now() - this.startTime) * 60000; // WPM

        return features;
    }

    processMousePatterns() {
        const mouseEvents = this.patterns.mouse;
        const features = {};

        if (mouseEvents.length === 0) return features;

        // Calculate mouse velocity and acceleration
        const movements = mouseEvents.filter(e => e.type === 'move');
        let totalDistance = 0;
        let velocities = [];
        let accelerations = [];

        for (let i = 1; i < movements.length; i++) {
            const prev = movements[i - 1];
            const curr = movements[i];

            const distance = Math.sqrt(Math.pow(curr.x - prev.x, 2) + Math.pow(curr.y - prev.y, 2));
            const time = curr.timestamp - prev.timestamp;
            const velocity = time > 0 ? distance / time : 0;

            totalDistance += distance;
            velocities.push(velocity);

            if (i > 1) {
                const acceleration = (velocities[i - 1] - velocities[i - 2]) / time;
                accelerations.push(acceleration);
            }
        }

        features.avgMouseSpeed = velocities.length > 0 ? velocities.reduce((a, b) => a + b, 0) / velocities.length : 0;
        features.avgMouseAcceleration = accelerations.length > 0 ? accelerations.reduce((a, b) => a + b, 0) / accelerations.length : 0;
        features.totalMouseDistance = totalDistance;
        features.mouseMovements = movements.length;

        // Click patterns
        const clicks = mouseEvents.filter(e => e.type === 'down');
        features.clickCount = clicks.length;

        return features;
    }

    processTouchPatterns() {
        const touchEvents = this.patterns.touch;
        const features = {};

        if (touchEvents.length === 0) return features;

        // Calculate swipe patterns
        const touchSessions = {};

        touchEvents.forEach(touch => {
            if (!touchSessions[touch.identifier]) {
                touchSessions[touch.identifier] = [];
            }
            touchSessions[touch.identifier].push(touch);
        });

        const swipeData = [];
        for (let sessionId in touchSessions) {
            const session = touchSessions[sessionId];
            if (session.length < 2) continue;

            const start = session[0];
            const end = session[session.length - 1];

            const swipe = {
                startX: start.x,
                startY: start.y,
                endX: end.x,
                endY: end.y,
                length: Math.sqrt(Math.pow(end.x - start.x, 2) + Math.pow(end.y - start.y, 2)),
                duration: end.timestamp - start.timestamp,
                avgPressure: session.reduce((sum, t) => sum + t.force, 0) / session.length,
                avgArea: session.reduce((sum, t) => sum + (t.radiusX * t.radiusY), 0) / session.length
            };

            swipeData.push(swipe);
        }

        features.swipeCount = swipeData.length;
        features.avgSwipeLength = swipeData.length > 0 ? swipeData.reduce((sum, s) => sum + s.length, 0) / swipeData.length : 0;
        features.avgSwipeDuration = swipeData.length > 0 ? swipeData.reduce((sum, s) => sum + s.duration, 0) / swipeData.length : 0;
        features.avgSwipePressure = swipeData.length > 0 ? swipeData.reduce((sum, s) => sum + s.avgPressure, 0) / swipeData.length : 0;

        return features;
    }

    // Send patterns to backend for processing
    async sendPatternsToServer(patterns, isRegistration = false) {
        try {
            const endpoint = isRegistration ? '/api/register' : '/api/login';
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    patterns: patterns,
                    timestamp: Date.now()
                })
            });

            return await response.json();
        } catch (error) {
            console.error('Error sending patterns:', error);
            return { success: false, error: error.message };
        }
    }
}
