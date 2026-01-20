import { useRef, useEffect, useState } from 'react';
import Webcam from 'react-webcam';
import { useAuth } from './AuthContext';
import { Camera, CheckCircle, AlertTriangle, Loader2, ArrowLeft } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

// WebSocket URL
const WS_URL = 'ws://localhost:8000/api/v1/ws/stream/mobile_cam';

export default function LiveCheckin() {
    const { user } = useAuth();
    const navigate = useNavigate();
    const webcamRef = useRef(null);

    const [status, setStatus] = useState('connecting'); // connecting, streaming, success, spoof, error
    const [message, setMessage] = useState('Đang kết nối server...');
    const [similarity, setSimilarity] = useState(0);

    const ws = useRef(null);
    const intervalRef = useRef(null);

    useEffect(() => {
        connectWebSocket();
        return () => {
            cleanup();
        };
    }, []);

    const cleanup = () => {
        if (ws.current) ws.current.close();
        if (intervalRef.current) clearInterval(intervalRef.current);
    };

    const connectWebSocket = () => {
        ws.current = new WebSocket(WS_URL);

        ws.current.onopen = () => {
            console.log('WS Connected');
            setStatus('streaming');
            setMessage('Đang tìm khuôn mặt...');
            // Start sending frames
            startSendingFrames();
        };

        ws.current.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.type === 'update') {
                processEvents(data.events);
            }
        };

        ws.current.onerror = (err) => {
            console.error('WS Error:', err);
            setStatus('error');
            setMessage('Lỗi kết nối WebSocket');
        };

        ws.current.onclose = () => {
            console.log('WS Closed');
            if (status !== 'success') {
                setStatus('disconnected');
                setMessage('Mất kết nối server');
            }
        };
    };

    const processEvents = (events) => {
        if (!events || events.length === 0) return;

        // Get the latest significant event
        // The backend StreamProcessor emits events when decisions are made
        const relevantEvents = events.filter(e => e.decision !== 'pending');

        if (relevantEvents.length > 0) {
            const latest = relevantEvents[relevantEvents.length - 1];

            if (latest.decision === 'accepted') {
                setStatus('success');
                setMessage(`Xin chào: ${latest.user_id}`);
                setSimilarity(latest.confidence);
                cleanup(); // Stop scanning on success
            } else if (latest.decision === 'spoof') {
                setStatus('spoof');
                setMessage('CẢNH BÁO GIẢ MẠO!');
            } else if (latest.decision === 'rejected') {
                setStatus('error');
                setMessage('Không nhận diện được khuôn mặt');
            }
        }
    };

    const startSendingFrames = () => {
        // Send frames at 10 FPS (100ms)
        intervalRef.current = setInterval(() => {
            if (ws.current && ws.current.readyState === WebSocket.OPEN && webcamRef.current) {
                const imageSrc = webcamRef.current.getScreenshot();
                if (imageSrc) {
                    // Remove "data:image/jpeg;base64," prefix
                    const base64Data = imageSrc.split(',')[1];
                    ws.current.send(JSON.stringify({
                        type: 'frame',
                        frame: base64Data
                    }));
                }
            }
        }, 100);
    };

    return (
        <div className="min-h-screen bg-black flex flex-col items-center justify-center text-white relative overflow-hidden">

            {/* Video Fullscreen */}
            <div className="absolute inset-0 z-0">
                <Webcam
                    ref={webcamRef}
                    audio={false}
                    screenshotFormat="image/jpeg"
                    videoConstraints={{ facingMode: "user" }}
                    className="w-full h-full object-cover opacity-60"
                />
            </div>

            {/* Overlay UI */}
            <div className="z-10 w-full max-w-md p-6 flex flex-col items-center">

                <h2 className="text-3xl font-bold mb-8 drop-shadow-md">Live Check-in</h2>

                {status === 'streaming' && (
                    <div className="bg-black/50 backdrop-blur-md p-8 rounded-3xl flex flex-col items-center gap-4 animate-pulse border border-white/10">
                        <div className="relative">
                            <div className="w-24 h-24 rounded-full border-t-4 border-blue-500 animate-spin"></div>
                            <Camera className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 text-blue-400" size={32} />
                        </div>
                        <p className="text-xl font-medium text-blue-200">{message}</p>
                    </div>
                )}

                {status === 'success' && (
                    <div className="bg-green-500/90 backdrop-blur-md p-8 rounded-3xl flex flex-col items-center gap-4 transform scale-110 transition-all shadow-2xl shadow-green-500/50">
                        <CheckCircle size={80} className="text-white drop-shadow-lg" />
                        <div className="text-center">
                            <h3 className="text-3xl font-bold">Thành Công!</h3>
                            <p className="text-xl mt-2">{message}</p>
                            <p className="text-sm opacity-80 mt-1">Độ chính xác: {(similarity * 100).toFixed(0)}%</p>
                        </div>
                    </div>
                )}

                {status === 'spoof' && (
                    <div className="bg-red-600/90 backdrop-blur-md p-8 rounded-3xl flex flex-col items-center gap-4 animate-bounce shadow-2xl shadow-red-600/50">
                        <AlertTriangle size={80} className="text-white" />
                        <h3 className="text-3xl font-bold">CẢNH BÁO</h3>
                        <p className="text-xl">{message}</p>
                    </div>
                )}

                {status === 'error' || status === 'disconnected' ? (
                    <div className="bg-gray-800/80 p-6 rounded-2xl flex flex-col items-center gap-2">
                        <AlertTriangle className="text-orange-500" size={40} />
                        <p>{message}</p>
                        <button onClick={() => window.location.reload()} className="bg-blue-600 px-4 py-2 rounded mt-2">Thử lại</button>
                    </div>
                ) : null}

                <button
                    onClick={() => navigate('/')}
                    className="absolute top-6 left-6 p-3 bg-white/10 hover:bg-white/20 rounded-full backdrop-blur-sm transition-all"
                >
                    <ArrowLeft size={24} />
                </button>

            </div>
        </div>
    );
}
