import { useRef, useState, useEffect } from 'react';
import Webcam from 'react-webcam';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import {
    Shield,
    Camera,
    Search,
    UserCheck,
    Lock,
    CheckCircle2,
    XCircle,
    AlertCircle,
    ArrowLeft,
    Loader2
} from 'lucide-react';
import { useAuth } from './AuthContext';

export default function SecureCheckin() {
    const { user } = useAuth();
    const navigate = useNavigate();
    const webcamRef = useRef(null);

    const [isProcessing, setIsProcessing] = useState(false);
    const [result, setResult] = useState(null);
    const [steps, setSteps] = useState([
        { id: 'detecting', label: 'Tìm khuôn mặt', status: 'idle', icon: Search },
        { id: 'anti_spoofing', label: 'Xác thực sinh trắc', status: 'idle', icon: Shield },
        { id: 'recognizing', label: 'Nhận diện danh tính', status: 'idle', icon: UserCheck },
    ]);

    const updateStep = (id, status, message = "") => {
        setSteps(prev => prev.map(step =>
            step.id === id ? { ...step, status, message } : step
        ));
    };

    const handleCheckin = async () => {
        if (!webcamRef.current) return;

        setIsProcessing(true);
        setResult(null);
        // Reset steps
        setSteps(prev => prev.map(s => ({ ...s, status: 'idle', message: '' })));

        try {
            const imageSrc = webcamRef.current.getScreenshot();
            if (!imageSrc) throw new Error("Không thể chụp ảnh từ webcam");

            const res = await fetch(imageSrc);
            const blob = await res.blob();
            const formData = new FormData();
            formData.append('file', blob, 'secure_checkin.jpg');
            formData.append('expected_user_id', user.face_user_id);

            // Start first step
            updateStep('detecting', 'processing');

            const response = await axios.post('/api/v1/checkin_fas', formData);
            const data = response.data;

            // Sync steps from API response
            data.steps.forEach(apiStep => {
                const statusMap = {
                    'success': 'success',
                    'failed': 'error',
                    'pending': 'processing',
                    'error': 'error',
                    'skipped': 'idle'
                };
                updateStep(apiStep.step_name, statusMap[apiStep.status] || 'idle', apiStep.message);
            });

            if (data.success) {
                setResult({
                    status: 'success',
                    name: data.name,
                    confidence: data.confidence,
                    fas_score: data.fas_score,
                    similarity: data.similarity
                });
            } else {
                setResult({
                    status: data.is_spoof ? 'spoof' : 'failed',
                    message: data.message
                });
            }

        } catch (err) {
            console.error(err);
            setResult({
                status: 'failed',
                message: err.response?.data?.detail || "Lỗi hệ thống không xác định"
            });
        } finally {
            setIsProcessing(false);
        }
    };

    return (
        <div className="min-h-screen bg-slate-950 text-white flex flex-col items-center p-6 font-sans">
            {/* Header */}
            <div className="w-full max-w-4xl flex items-center justify-between mb-8 z-10">
                <button
                    onClick={() => navigate('/')}
                    className="p-3 bg-white/5 hover:bg-white/10 rounded-2xl transition-all border border-white/5"
                >
                    <ArrowLeft size={24} />
                </button>
                <div className="flex flex-col items-center">
                    <div className="flex items-center gap-2 text-blue-400 mb-1">
                        <Lock size={18} />
                        <span className="text-xs font-bold uppercase tracking-widest">Secure Matrix v2</span>
                    </div>
                    <h1 className="text-2xl font-black bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent">
                        XÁC THỰC BẢO MẬT
                    </h1>
                </div>
                <div className="w-12" /> {/* Spacer */}
            </div>

            <div className="w-full max-w-5xl grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">

                {/* Left Side: Camera & Result */}
                <div className="flex flex-col gap-6">
                    <div className="relative rounded-[2.5rem] overflow-hidden border-4 border-slate-800 shadow-2xl bg-black aspect-[4/3]">
                        <Webcam
                            ref={webcamRef}
                            audio={false}
                            screenshotFormat="image/jpeg"
                            videoConstraints={{ facingMode: "user" }}
                            className={`w-full h-full object-cover transition-opacity duration-700 ${isProcessing ? 'opacity-40' : 'opacity-100'}`}
                        />

                        {/* Scanning Animation */}
                        {isProcessing && (
                            <div className="absolute inset-0 z-20 overflow-hidden pointer-events-none">
                                <div className="w-full h-1 bg-blue-500/50 absolute top-0 animate-scan shadow-[0_0_20px_rgba(59,130,246,0.8)]"></div>
                                <div className="absolute inset-0 bg-blue-500/5 animate-pulse"></div>
                            </div>
                        )}

                        {/* Result Overlays */}
                        {result?.status === 'success' && (
                            <div className="absolute inset-0 bg-green-500/20 backdrop-blur-[2px] z-30 flex items-center justify-center animate-in fade-in zoom-in duration-300">
                                <div className="bg-slate-900/90 p-8 rounded-[2rem] border border-green-500/30 text-center shadow-2xl flex flex-col items-center">
                                    <CheckCircle2 size={64} className="text-green-500 mb-4 animate-bounce" />
                                    <h3 className="text-2xl font-bold text-white mb-2">Thành Công!</h3>
                                    <p className="text-slate-400 text-sm">Chào mừng quay trở lại,</p>
                                    <p className="text-xl font-black text-green-400">{result.name}</p>
                                </div>
                            </div>
                        )}

                        {result?.status === 'spoof' && (
                            <div className="absolute inset-0 bg-red-500/20 backdrop-blur-[2px] z-30 flex items-center justify-center animate-in fade-in zoom-in duration-300">
                                <div className="bg-slate-900/90 p-8 rounded-[2rem] border border-red-500/30 text-center shadow-2xl flex flex-col items-center">
                                    <AlertCircle size={64} className="text-red-500 mb-4 animate-pulse" />
                                    <h3 className="text-2xl font-bold text-white mb-2">GIẢ MẠO!</h3>
                                    <p className="text-red-400 font-medium">Hệ thống đã phát hiện gian lận sinh trắc.</p>
                                </div>
                            </div>
                        )}
                    </div>

                    {!isProcessing && !result && (
                        <button
                            onClick={handleCheckin}
                            className="w-full py-6 rounded-3xl bg-blue-600 hover:bg-blue-500 text-white font-black text-xl flex items-center justify-center gap-3 transition-all shadow-[0_0_40px_rgba(37,99,235,0.3)] hover:shadow-[0_0_50px_rgba(37,99,235,0.5)] active:scale-95"
                        >
                            <Camera size={28} />
                            QUÉT NGAY
                        </button>
                    )}

                    {result && (
                        <button
                            onClick={() => setResult(null)}
                            className="w-full py-6 rounded-3xl bg-slate-800 hover:bg-slate-700 text-white font-bold text-xl transition-all border border-white/5"
                        >
                            THỬ LẠI
                        </button>
                    )}
                </div>

                {/* Right Side: Process Steps */}
                <div className="bg-slate-900/50 backdrop-blur-xl rounded-[2.5rem] p-8 border border-white/5 shadow-2xl">
                    <h3 className="text-lg font-bold text-slate-400 uppercase tracking-widest mb-8 flex items-center gap-2">
                        <Loader2 className={isProcessing ? "animate-spin" : ""} size={20} />
                        Trạng thái quy trình
                    </h3>

                    <div className="flex flex-col gap-6">
                        {steps.map((step, idx) => {
                            const Icon = step.icon;
                            let statusColor = "text-slate-600 grayscale";
                            let bgColor = "bg-slate-800/50";
                            let iconColor = "text-slate-500";

                            if (step.status === 'processing') {
                                statusColor = "text-blue-400";
                                bgColor = "bg-blue-500/10 border border-blue-500/20";
                                iconColor = "text-blue-400 anime-pulse";
                            } else if (step.status === 'success') {
                                statusColor = "text-green-400";
                                bgColor = "bg-green-500/10 border border-green-500/20";
                                iconColor = "text-green-500";
                            } else if (step.status === 'error') {
                                statusColor = "text-red-400";
                                bgColor = "bg-red-500/10 border border-red-500/20";
                                iconColor = "text-red-500";
                            }

                            return (
                                <div key={step.id} className={`p-6 rounded-3xl flex items-center gap-5 transition-all duration-500 ${bgColor}`}>
                                    <div className={`p-4 rounded-2xl bg-black/40 ${iconColor}`}>
                                        {step.status === 'success' ? <CheckCircle2 size={28} /> :
                                            step.id === 'anti_spoofing' && result?.status === 'spoof' ? <XCircle size={28} /> :
                                                step.status === 'error' ? <XCircle size={28} /> :
                                                    <Icon size={28} className={step.status === 'processing' ? "animate-pulse" : ""} />}
                                    </div>
                                    <div className="flex-1">
                                        <div className="flex justify-between items-center mb-1">
                                            <span className={`font-bold text-lg ${statusColor}`}>{step.label}</span>
                                            {step.status === 'processing' && <Loader2 size={18} className="animate-spin text-blue-400" />}
                                        </div>
                                        <p className="text-sm text-slate-500 font-medium">
                                            {step.message || (step.status === 'idle' ? 'Đang chờ...' : '')}
                                        </p>
                                    </div>
                                </div>
                            );
                        })}
                    </div>

                    {result?.status === 'success' && (
                        <div className="mt-8 p-6 rounded-3xl bg-green-500/5 border border-green-500/10 grid grid-cols-2 gap-4">
                            <div className="text-center">
                                <p className="text-[10px] text-slate-500 uppercase font-bold mb-1">FAS Reliability</p>
                                <p className="text-lg font-black text-green-400">{(result.fas_score * 100).toFixed(1)}%</p>
                            </div>
                            <div className="text-center">
                                <p className="text-[10px] text-slate-500 uppercase font-bold mb-1">Match Confidence</p>
                                <p className="text-lg font-black text-blue-400">{(result.similarity * 100).toFixed(1)}%</p>
                            </div>
                        </div>
                    )}
                </div>

            </div>

            {/* Background Decorations */}
            <div className="fixed bottom-0 left-0 w-96 h-96 bg-blue-600/10 blur-[120px] -z-10 rounded-full"></div>
            <div className="fixed top-0 right-0 w-96 h-96 bg-purple-600/10 blur-[120px] -z-10 rounded-full"></div>
        </div>
    );
}
