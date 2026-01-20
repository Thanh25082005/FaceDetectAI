import { useState, useRef, useEffect, useCallback } from 'react'
import Webcam from 'react-webcam'
import axios from 'axios'
import { MapPin, CheckCircle, AlertTriangle, Loader2, LogOut, Camera, RefreshCcw } from 'lucide-react'
import { useAuth } from './AuthContext'
import { useNavigate } from 'react-router-dom'

export default function Dashboard() {
    const { user, logout } = useAuth();
    const navigate = useNavigate();

    const webcamRef = useRef(null)
    const canvasRef = useRef(null)

    const [location, setLocation] = useState(null)
    const [error, setError] = useState(null)
    const [isScanning, setIsScanning] = useState(false)
    const [scanResult, setScanResult] = useState({ status: 'idle', message: '' }) // idle, success, error, spoof
    const [distance, setDistance] = useState(null)

    const scanIntervalRef = useRef(null)

    // 1. Get Location
    useEffect(() => {
        if (!navigator.geolocation) {
            setError("Geolocation is not supported")
            return
        }
        navigator.geolocation.getCurrentPosition(
            (pos) => setLocation({ latitude: pos.coords.latitude, longitude: pos.coords.longitude }),
            (err) => setError("Vui lòng bật GPS để chấm công"),
            { enableHighAccuracy: true }
        )
    }, [])

    const handleLogout = () => {
        logout();
        navigate('/login');
    }

    // 2. Draw Box Helper
    const drawBox = (box, color = 'blue') => {
        const canvas = canvasRef.current;
        if (!canvas || !webcamRef.current) return;

        // Match canvas size to video size
        const video = webcamRef.current.video;
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (box) {
            const [x1, y1, x2, y2] = box;
            const width = x2 - x1;
            const height = y2 - y1;

            ctx.strokeStyle = color;
            ctx.lineWidth = 4;
            ctx.strokeRect(x1, y1, width, height);

            // Add label
            ctx.fillStyle = color;
            ctx.fillRect(x1, y1 - 25, width, 25);
            ctx.fillStyle = 'white';
            ctx.font = '16px sans-serif';
            ctx.fillText(scanResult.status === 'success' ? 'MATCHED' : 'SCANNING', x1 + 5, y1 - 5);
        }
    }

    // 3. Process Frame
    const processFrame = async () => {
        if (!webcamRef.current || !location) return;

        try {
            const imageSrc = webcamRef.current.getScreenshot();
            if (!imageSrc) return;

            const res = await fetch(imageSrc);
            const blob = await res.blob();

            const formData = new FormData();
            formData.append('file', blob, 'checkin.jpg');
            // Call API - Switching to /recognize_face as requested
            // Note: This endpoint does not support Geolocation or spoofing check
            const response = await axios.post('/api/v1/recognize_face?threshold=0.5', formData);
            const data = response.data;

            let matched = false;
            let box = null;
            let msg = '';
            let isSpoof = false;

            if (data.success && data.faces_detected > 0) {
                // Check matches
                if (data.matches && data.matches.length > 0) {
                    const bestMatch = data.matches[0];
                    box = bestMatch.box;

                    if (bestMatch.is_match) {
                        // Check if matches logged-in user
                        if (bestMatch.user_id === user.face_user_id) {
                            matched = true;
                            msg = `Chấm công thành công: ${bestMatch.name || bestMatch.user_id}`;
                        } else {
                            msg = `Không khớp: ${bestMatch.name || bestMatch.user_id} != ${user.face_user_id}`;
                        }
                    } else {
                        msg = "Không nhận diện được";
                    }
                } else {
                    msg = "Đã phát hiện mặt nhưng không nhận diện được";
                }
            } else {
                msg = "Không tìm thấy khuôn mặt";
            }

            // Visual Feedback
            if (box) {
                let color = '#3b82f6'; // blue
                if (matched) color = '#22c55e'; // green
                drawBox(box, color);
            } else {
                // Clear canvas if no face
                const canvas = canvasRef.current;
                if (canvas) {
                    const ctx = canvas.getContext('2d');
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                }
            }

            // Handle Result
            if (matched) {
                setScanResult({ status: 'success', message: msg });
                stopScan(); // AUTO STOP on Success
            } else {
                setScanResult({ status: 'scanning', message: msg });
            }

        } catch (err) {
            console.error(err);
            setScanResult({ status: 'error', message: 'Lỗi kết nối hoặc xử lý' });
        }
    };

    // 4. Start/Stop Scan Loop
    const startScan = () => {
        setIsScanning(true);
        setScanResult({ status: 'scanning', message: 'Đang quét...' });
        // Scan every 500ms
        scanIntervalRef.current = setInterval(processFrame, 500);
    };

    const stopScan = () => {
        setIsScanning(false);
        if (scanIntervalRef.current) {
            clearInterval(scanIntervalRef.current);
            scanIntervalRef.current = null;
        }
    };

    // Cleanup
    useEffect(() => {
        return () => stopScan();
    }, []);

    return (
        <div className="min-h-screen bg-gray-50 flex flex-col items-center p-4">
            {/* Header */}
            <div className="w-full max-w-md flex justify-between items-center mb-6 pt-4">
                <div>
                    <h2 className="text-xl font-bold text-gray-800">Xin chào, {user?.full_name}</h2>
                    <p className="text-xs text-gray-500">ID: {user?.face_user_id}</p>
                </div>
                <button onClick={handleLogout} className="p-2 bg-white rounded-full shadow text-gray-600 hover:text-red-500">
                    <LogOut size={20} />
                </button>
            </div>

            <div className="w-full max-w-md bg-white rounded-2xl shadow-xl overflow-hidden p-6 text-center">
                <h1 className="text-2xl font-bold mb-4 text-blue-600">Chấm công Face ID</h1>

                {/* Camera Area with Canvas Overlay */}
                <div className="aspect-[3/4] bg-black rounded-xl overflow-hidden relative mb-6">
                    <Webcam
                        audio={false}
                        ref={webcamRef}
                        screenshotFormat="image/jpeg"
                        videoConstraints={{ facingMode: "user" }}
                        className="w-full h-full object-cover"
                    />
                    <canvas
                        ref={canvasRef}
                        className="absolute inset-0 w-full h-full pointer-events-none"
                    />
                </div>

                {/* Status Messages for Pipeline */}
                {error && <div className="text-red-500 text-sm mb-4 bg-red-50 p-2 rounded">{error}</div>}

                {scanResult.status === 'success' && (
                    <div className="bg-green-100 text-green-700 p-3 rounded-lg mb-4 animate-bounce-short">
                        <div className="flex justify-center items-center gap-2 font-bold"><CheckCircle size={18} /> {scanResult.message}</div>
                        {distance && <p className="text-xs mt-1">Khoảng cách: {distance.toFixed(0)}m</p>}
                    </div>
                )}

                {scanResult.status === 'spoof' && (
                    <div className="bg-red-100 text-red-700 p-3 rounded-lg mb-4 font-bold border border-red-200">
                        <div className="flex justify-center items-center gap-2"><AlertTriangle size={18} /> {scanResult.message}</div>
                    </div>
                )}

                {isScanning && scanResult.status === 'scanning' && (
                    <div className="bg-blue-50 text-blue-600 p-2 rounded-lg mb-4 text-sm flex items-center justify-center gap-2">
                        <Loader2 className="animate-spin" size={16} />
                        {scanResult.message}
                    </div>
                )}


                {/* Action Button */}
                {!isScanning && scanResult.status !== 'success' && (
                    <button
                        onClick={startScan}
                        disabled={!location}
                        className={`w-full py-4 text-white text-lg font-bold rounded-xl flex items-center justify-center gap-2 transition-all
               ${location ? 'bg-blue-600 hover:bg-blue-700 shadow-lg shadow-blue-200' : 'bg-gray-300 cursor-not-allowed'}
             `}
                    >
                        <Camera className={location ? "animate-pulse" : ""} />
                        {location ? "BẮT ĐẦU QUÉT" : "Đang lấy vị trí..."}
                    </button>
                )}

                {isScanning && (
                    <button
                        onClick={stopScan}
                        className="w-full py-4 bg-red-500 hover:bg-red-600 text-white text-lg font-bold rounded-xl flex items-center justify-center gap-2"
                    >
                        <AlertTriangle /> Dừng quét
                    </button>
                )}

                {/* Live Mode Shortcut */}
                {!isScanning && (
                    <button
                        onClick={() => navigate('/live')}
                        className="w-full mt-4 py-3 bg-purple-600 hover:bg-purple-700 text-white font-bold rounded-xl flex items-center justify-center gap-2"
                    >
                        ⚡ Chế độ Live Stream
                    </button>
                )}

                {scanResult.status === 'success' && (
                    <button onClick={() => { setScanResult({ status: 'idle', message: '' }); startScan(); }}
                        className="w-full py-4 bg-gray-200 hover:bg-gray-300 text-gray-700 font-bold rounded-xl flex items-center justify-center gap-2">
                        <RefreshCcw size={18} /> Chấm công lại
                    </button>
                )}

            </div>
        </div>
    )
}
