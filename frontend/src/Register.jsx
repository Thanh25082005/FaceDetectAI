import { useState, useRef } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import Webcam from 'react-webcam';
import axios from 'axios';
import { User, Lock, Calendar, Camera, ArrowRight, Loader2, CheckCircle, AlertCircle } from 'lucide-react';

export default function Register() {
    const navigate = useNavigate();
    const webcamRef = useRef(null);

    const [step, setStep] = useState(1); // 1: Info, 2: Face Scan
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const [formData, setFormData] = useState({
        username: '',
        password: '',
        full_name: '',
        dob: ''
    });

    const [faceUserId, setFaceUserId] = useState(null);

    const handleRegister = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError('');

        try {
            // Step 1: Create Account
            const res = await axios.post('/api/v1/auth/register', formData);
            if (res.data.success) {
                setFaceUserId(res.data.face_user_id);
                setStep(2); // Move to Face Scan
            }
        } catch (err) {
            setError(err.response?.data?.detail || "Registration failed");
        } finally {
            setLoading(false);
        }
    };

    const handleFaceScan = async () => {
        const imageSrc = webcamRef.current.getScreenshot();
        if (!imageSrc) return;

        setLoading(true);
        try {
            const res = await fetch(imageSrc);
            const blob = await res.blob();

            const data = new FormData();
            data.append('file', blob, 'face.jpg');
            data.append('user_id', faceUserId);
            data.append('name', formData.full_name);

            await axios.post('/api/v1/add_face', data);

            // Success! Redirect to login
            navigate('/login');

        } catch (err) {
            setError("Không tìm thấy khuôn mặt hoặc ảnh mờ. Vui lòng thử lại.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-gray-50 flex items-center justify-center p-4">
            <div className="w-full max-w-md bg-white rounded-2xl shadow-lg p-6">

                <h1 className="text-2xl font-bold text-center text-gray-800 mb-2">Đăng ký tài khoản</h1>
                <div className="flex justify-center mb-6">
                    <div className={`h-2 w-16 rounded mr-2 ${step >= 1 ? 'bg-blue-600' : 'bg-gray-200'}`}></div>
                    <div className={`h-2 w-16 rounded ${step >= 2 ? 'bg-blue-600' : 'bg-gray-200'}`}></div>
                </div>

                {error && (
                    <div className="bg-red-50 text-red-600 p-3 rounded-lg text-sm mb-4 flex items-center gap-2">
                        <AlertCircle size={16} /> {error}
                    </div>
                )}

                {step === 1 ? (
                    <form onSubmit={handleRegister} className="space-y-4">
                        <div>
                            <label className="text-sm font-medium text-gray-700">Họ và tên</label>
                            <div className="relative mt-1">
                                <User className="absolute left-3 top-3 text-gray-400" size={18} />
                                <input
                                    required
                                    className="w-full pl-10 pr-4 py-2 border rounded-xl outline-none focus:ring-2 focus:ring-blue-500"
                                    placeholder="Nguyễn Văn A"
                                    value={formData.full_name}
                                    onChange={e => setFormData({ ...formData, full_name: e.target.value })}
                                />
                            </div>
                        </div>

                        <div>
                            <label className="text-sm font-medium text-gray-700">Ngày sinh</label>
                            <div className="relative mt-1">
                                <Calendar className="absolute left-3 top-3 text-gray-400" size={18} />
                                <input
                                    type="date"
                                    required
                                    className="w-full pl-10 pr-4 py-2 border rounded-xl outline-none focus:ring-2 focus:ring-blue-500"
                                    value={formData.dob}
                                    onChange={e => setFormData({ ...formData, dob: e.target.value })}
                                />
                            </div>
                        </div>

                        <div>
                            <label className="text-sm font-medium text-gray-700">Tên đăng nhập</label>
                            <input
                                required
                                className="w-full mt-1 px-4 py-2 border rounded-xl outline-none focus:ring-2 focus:ring-blue-500"
                                placeholder="username"
                                value={formData.username}
                                onChange={e => setFormData({ ...formData, username: e.target.value })}
                            />
                        </div>

                        <div>
                            <label className="text-sm font-medium text-gray-700">Mật khẩu</label>
                            <div className="relative mt-1">
                                <Lock className="absolute left-3 top-3 text-gray-400" size={18} />
                                <input
                                    type="password"
                                    required
                                    className="w-full pl-10 pr-4 py-2 border rounded-xl outline-none focus:ring-2 focus:ring-blue-500"
                                    placeholder="******"
                                    value={formData.password}
                                    onChange={e => setFormData({ ...formData, password: e.target.value })}
                                />
                            </div>
                        </div>

                        <button
                            type="submit"
                            disabled={loading}
                            className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 rounded-xl flex items-center justify-center gap-2 mt-4"
                        >
                            {loading ? <Loader2 className="animate-spin" /> : <>Tiếp tục <ArrowRight size={18} /></>}
                        </button>

                        <p className="mt-4 text-center text-sm text-gray-600">
                            Đã có tài khoản? <Link to="/login" className="text-blue-600 font-bold hover:underline">Đăng nhập</Link>
                        </p>
                    </form>
                ) : (
                    <div className="text-center">
                        <p className="mb-4 text-gray-600">Vui lòng nhìn thẳng vào camera để đăng ký khuôn mặt.</p>

                        <div className="aspect-[3/4] bg-black rounded-lg overflow-hidden relative mb-4 mx-auto max-w-xs ring-4 ring-blue-100">
                            <Webcam
                                audio={false}
                                ref={webcamRef}
                                screenshotFormat="image/jpeg"
                                videoConstraints={{ facingMode: "user" }}
                                className="w-full h-full object-cover"
                            />
                        </div>

                        <button
                            onClick={handleFaceScan}
                            disabled={loading}
                            className="w-full bg-green-600 hover:bg-green-700 text-white font-bold py-3 rounded-xl flex items-center justify-center gap-2"
                        >
                            {loading ? <Loader2 className="animate-spin" /> : <><Camera size={18} /> Chụp & Hoàn tất</>}
                        </button>
                    </div>
                )}
            </div>
        </div>
    );
}
