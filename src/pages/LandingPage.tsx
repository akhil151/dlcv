import React, { useRef, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';

export const LandingPage: React.FC = () => {
  const navigate = useNavigate();
  const demoSectionRef = useRef<HTMLDivElement>(null);
  const [salesModalOpen, setSalesModalOpen] = useState(false);
  const [apiModalOpen, setApiModalOpen] = useState(false);

  const scrollToDemo = () => {
    demoSectionRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <div className="font-body-md overflow-x-hidden min-h-screen bg-surface text-on-surface flex flex-col selection:bg-primary/30 selection:text-primary">
      {/* TopNavBar */}
      <header className="fixed top-0 left-0 w-full z-50 flex justify-between items-center px-gutter h-16 bg-surface/80 backdrop-blur-md border-b border-outline-variant">
        <div className="flex items-center gap-md">
          <Link to="/" className="font-display text-headline-md font-bold text-on-surface">
            SafeFrame
          </Link>
          <nav className="hidden md:flex items-center gap-lg ml-xl">
            <Link
              to="/dashboard"
              className="text-primary dark:text-primary font-bold border-b-2 border-primary py-sm font-label-md text-label-md"
            >
              Dashboard
            </Link>
            <Link
              to="/upload"
              className="text-on-surface-variant dark:text-on-surface-variant font-label-md text-label-md hover:text-primary transition-colors duration-200"
            >
              Upload
            </Link>
          </nav>
        </div>
        <div className="flex items-center gap-md">
          <button
            type="button"
            onClick={() => setApiModalOpen(true)}
            className="material-symbols-outlined text-on-surface-variant hover:text-primary cursor-pointer transition-colors"
            title="SafeFrame Developer API & SDK"
          >
            code
          </button>
          <button
            type="button"
            onClick={() => navigate('/upload')}
            className="bg-primary-container text-on-primary-container px-lg py-sm rounded-lg font-label-md text-label-md font-bold hover:opacity-80 active:scale-95 transition-all cursor-pointer"
          >
            Get Started
          </button>
        </div>
      </header>

      {/* Main Content Stage */}
      <main className="pt-16 flex-1 flex flex-col">
        {/* Hero Section */}
        <section className="relative min-h-[870px] flex items-center overflow-hidden px-gutter hero-gradient">
          <div className="absolute inset-0 noise-bg pointer-events-none"></div>
          <div className="max-w-container-max mx-auto w-full grid grid-cols-1 lg:grid-cols-2 gap-xl items-center py-xl">
            <div className="flex flex-col gap-lg z-10">
              <div className="inline-flex items-center gap-sm bg-primary/10 text-primary border border-primary/20 px-md py-xs rounded-full w-fit">
                <span className="material-symbols-outlined text-[16px]">verified_user</span>
                <span className="font-label-sm text-label-sm">Next-Gen AI Redaction Engine</span>
              </div>
              <h1 className="font-display text-display leading-tight text-on-surface max-w-xl">
                Protect Privacy in Every Frame
              </h1>
              <p className="font-body-lg text-body-lg text-on-surface-variant max-w-lg">
                Automatically detect and blur faces, screens, license plates, and ID cards using AI-powered object detection and tracking. Precise, fast, and entirely invisible.
              </p>
              <div className="flex flex-wrap gap-md mt-sm">
                <button
                  type="button"
                  onClick={() => navigate('/upload')}
                  className="bg-primary-container text-on-primary-container px-xl py-md rounded-lg font-label-md text-label-md font-bold hover:opacity-90 active:scale-95 transition-all flex items-center gap-sm cursor-pointer"
                >
                  <span className="material-symbols-outlined">upload_file</span>
                  <span>Upload Video</span>
                </button>
                <button
                  type="button"
                  onClick={scrollToDemo}
                  className="border border-outline-variant text-on-surface px-xl py-md rounded-lg font-label-md text-label-md font-bold hover:bg-surface-variant active:scale-95 transition-all flex items-center gap-sm cursor-pointer"
                >
                  <span className="material-symbols-outlined">play_circle</span>
                  <span>Watch Demo</span>
                </button>
              </div>
            </div>

            {/* Hero Illustration: Mockup Player */}
            <div className="relative z-10 group" ref={demoSectionRef}>
              <div className="absolute -inset-4 bg-primary/20 blur-[100px] rounded-full opacity-30 group-hover:opacity-50 transition-opacity"></div>
              <div className="glass-card rounded-xl overflow-hidden shadow-2xl relative border border-outline-variant/50">
                <div className="h-10 bg-surface-container flex items-center px-md gap-sm border-b border-outline-variant/30">
                  <div className="flex gap-xs">
                    <div className="w-3 h-3 rounded-full bg-error/40"></div>
                    <div className="w-3 h-3 rounded-full bg-tertiary/40"></div>
                    <div className="w-3 h-3 rounded-full bg-primary/40"></div>
                  </div>
                  <div className="bg-surface-container-low px-md py-xs rounded text-[10px] text-on-surface-variant font-mono">
                    player_v2.0_stable
                  </div>
                </div>
                <div className="aspect-video relative bg-black">
                  <div
                    className="absolute inset-0 bg-cover bg-center"
                    style={{
                      backgroundImage: `url('https://lh3.googleusercontent.com/aida-public/AB6AXuDbW2EWDydDRNtsyANgQANIbVdm3kkdR5vVHCMklKrDmnKZ4yZpFNh_kaMz9uPVJ9GmFccDufq36wBKDtcIrjTwAoXmks2P-l8j9j20OnsBcOi1xQ151jhCLi4ATiPy9NmY6svEajYVl3uLZqTJ0Ylzqakd-SKgKfikALQDmCnKQ7ZlOt18_iaWxoXgGaI1A8CZ2Yujd8nenUA_8Uh94SUzjKBp8Kkhc3RRWFkRFoHktPTqk-zBWj0y7cvxmVYmIjmEjVYLu_ezhZc')`,
                    }}
                  ></div>
                  {/* Redaction Overlays */}
                  <div className="absolute top-[20%] left-[30%] w-24 h-24 border-2 border-primary active-redaction flex items-center justify-center">
                    <div className="w-full h-full backdrop-blur-3xl opacity-80"></div>
                    <span className="absolute top-0 left-0 -translate-y-full bg-primary text-on-primary text-[10px] px-sm py-xs font-bold uppercase">
                      Person #24
                    </span>
                  </div>
                  <div className="absolute bottom-[25%] right-[15%] w-32 h-16 border-2 border-primary/50 active-redaction flex items-center justify-center">
                    <div className="w-full h-full backdrop-blur-2xl opacity-90"></div>
                    <span className="absolute top-0 left-0 -translate-y-full bg-primary/80 text-on-primary text-[10px] px-sm py-xs font-bold uppercase">
                      Screen
                    </span>
                  </div>
                  {/* Player Controls */}
                  <div className="absolute bottom-0 left-0 w-full p-md bg-gradient-to-t from-black to-transparent">
                    <div className="timeline-track mb-sm">
                      <div className="timeline-progress"></div>
                      <div className="timeline-tick" style={{ left: '10%' }}></div>
                      <div className="timeline-tick" style={{ left: '25%' }}></div>
                      <div className="timeline-tick" style={{ left: '45%' }}></div>
                      <div className="timeline-tick" style={{ left: '80%' }}></div>
                    </div>
                    <div className="flex justify-between items-center text-on-surface">
                      <div className="flex gap-md items-center">
                        <span className="material-symbols-outlined text-[20px]">pause</span>
                        <span className="material-symbols-outlined text-[20px]">volume_up</span>
                        <span className="font-label-sm text-label-sm">00:42 / 01:20</span>
                      </div>
                      <span className="material-symbols-outlined text-[20px]">fullscreen</span>
                    </div>
                  </div>
                </div>
              </div>
              {/* Floating Chips */}
              <div className="absolute -right-8 top-1/4 glass-card p-sm rounded-lg flex items-center gap-sm animate-bounce shadow-xl">
                <div className="w-2 h-2 bg-primary rounded-full"></div>
                <span className="font-label-sm text-label-sm">99.8% Confidence</span>
              </div>
            </div>
          </div>
        </section>

        {/* Workflow Section */}
        <section className="py-xl px-gutter bg-surface-container-lowest">
          <div className="max-w-container-max mx-auto">
            <div className="text-center mb-xl">
              <h2 className="font-display text-headline-lg text-on-surface">How SafeFrame Works</h2>
              <p className="font-body-md text-on-surface-variant mt-sm">
                Seamless technical precision from upload to delivery.
              </p>
            </div>
            <div className="flex flex-col md:flex-row justify-between items-center md:items-start gap-lg relative">
              {/* Connector Line */}
              <div className="hidden md:block absolute top-12 left-12 right-12 h-[1px] bg-outline-variant z-0"></div>
              {/* Steps */}
              <div className="flex flex-col items-center text-center gap-md z-10 max-w-[180px]">
                <div className="w-16 h-16 rounded-full bg-surface-container border border-outline-variant flex items-center justify-center text-primary shadow-lg">
                  <span className="material-symbols-outlined text-[32px]">upload</span>
                </div>
                <div>
                  <h4 className="font-label-md text-label-md text-on-surface font-bold">Upload</h4>
                  <p className="font-label-sm text-label-sm text-on-surface-variant mt-xs">
                    Drop your raw footage in 4K/60fps.
                  </p>
                </div>
              </div>
              <div className="flex flex-col items-center text-center gap-md z-10 max-w-[180px]">
                <div className="w-16 h-16 rounded-full bg-surface-container border border-outline-variant flex items-center justify-center text-primary shadow-lg">
                  <span className="material-symbols-outlined text-[32px]">psychology</span>
                </div>
                <div>
                  <h4 className="font-label-md text-label-md text-on-surface font-bold">AI Detection</h4>
                  <p className="font-label-sm text-label-sm text-on-surface-variant mt-xs">
                    Deep neural net identifies targets.
                  </p>
                </div>
              </div>
              <div className="flex flex-col items-center text-center gap-md z-10 max-w-[180px]">
                <div className="w-16 h-16 rounded-full bg-surface-container border border-outline-variant flex items-center justify-center text-primary shadow-lg">
                  <span className="material-symbols-outlined text-[32px]">polyline</span>
                </div>
                <div>
                  <h4 className="font-label-md text-label-md text-on-surface font-bold">Tracking</h4>
                  <p className="font-label-sm text-label-sm text-on-surface-variant mt-xs">
                    Objects tracked across occlusions.
                  </p>
                </div>
              </div>
              <div className="flex flex-col items-center text-center gap-md z-10 max-w-[180px]">
                <div className="w-16 h-16 rounded-full bg-surface-container border border-outline-variant flex items-center justify-center text-primary shadow-lg">
                  <span className="material-symbols-outlined text-[32px]">blur_on</span>
                </div>
                <div>
                  <h4 className="font-label-md text-label-md text-on-surface font-bold">Redaction</h4>
                  <p className="font-label-sm text-label-sm text-on-surface-variant mt-xs">
                    Dynamic blurring or masking applied.
                  </p>
                </div>
              </div>
              <div className="flex flex-col items-center text-center gap-md z-10 max-w-[180px]">
                <div className="w-16 h-16 rounded-full bg-primary border border-primary flex items-center justify-center text-on-primary shadow-lg shadow-primary/20">
                  <span className="material-symbols-outlined text-[32px]">download</span>
                </div>
                <div>
                  <h4 className="font-label-md text-label-md text-on-surface font-bold">Download</h4>
                  <p className="font-label-sm text-label-sm text-on-surface-variant mt-xs">
                    Secure export in original resolution.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Features Grid Section */}
        <section className="py-xl px-gutter">
          <div className="max-w-container-max mx-auto">
            <div className="flex flex-col md:flex-row justify-between items-end gap-md mb-xl">
              <div className="max-w-xl">
                <h2 className="font-display text-headline-lg text-on-surface">Precision-Grade Capabilities</h2>
                <p className="font-body-md text-on-surface-variant mt-sm">
                  SafeFrame leverages state-of-the-art transformer models to ensure no PII (Personally Identifiable Information) escapes the frame.
                </p>
              </div>
              <div className="bg-surface-container-high p-sm rounded-lg flex items-center gap-sm">
                <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
                <span className="font-label-sm text-label-sm">System Status: Optimal</span>
              </div>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-md">
              <div className="glass-card p-lg rounded-xl hover:bg-surface-variant/30 transition-all group">
                <span className="material-symbols-outlined text-primary text-[32px] mb-md group-hover:scale-110 transition-transform">
                  face
                </span>
                <h3 className="font-headline-md text-headline-md text-on-surface mb-sm">Face Detection</h3>
                <p className="font-body-md text-body-md text-on-surface-variant">
                  Multi-angle facial recognition with robust performance across lighting conditions.
                </p>
              </div>
              <div className="glass-card p-lg rounded-xl hover:bg-surface-variant/30 transition-all group">
                <span className="material-symbols-outlined text-primary text-[32px] mb-md group-hover:scale-110 transition-transform">
                  directions_car
                </span>
                <h3 className="font-headline-md text-headline-md text-on-surface mb-sm">License Plates</h3>
                <p className="font-body-md text-body-md text-on-surface-variant">
                  Automated masking for vehicle identification numbers across all global plate formats.
                </p>
              </div>
              <div className="glass-card p-lg rounded-xl hover:bg-surface-variant/30 transition-all group">
                <span className="material-symbols-outlined text-primary text-[32px] mb-md group-hover:scale-110 transition-transform">
                  monitor
                </span>
                <h3 className="font-headline-md text-headline-md text-on-surface mb-sm">Screen Detection</h3>
                <p className="font-body-md text-body-md text-on-surface-variant">
                  Identifies monitors, laptops, and mobile displays to prevent data leakage in video.
                </p>
              </div>
              <div className="glass-card p-lg rounded-xl hover:bg-surface-variant/30 transition-all group">
                <span className="material-symbols-outlined text-primary text-[32px] mb-md group-hover:scale-110 transition-transform">
                  badge
                </span>
                <h3 className="font-headline-md text-headline-md text-on-surface mb-sm">ID Card Detection</h3>
                <p className="font-body-md text-body-md text-on-surface-variant">
                  Detects and blurs ID cards, passports, and credit cards held in hand.
                </p>
              </div>
              <div className="glass-card p-lg rounded-xl hover:bg-surface-variant/30 transition-all group">
                <span className="material-symbols-outlined text-primary text-[32px] mb-md group-hover:scale-110 transition-transform">
                  visibility_off
                </span>
                <h3 className="font-headline-md text-headline-md text-on-surface mb-sm">Occlusion Recovery</h3>
                <p className="font-body-md text-body-md text-on-surface-variant">
                  Maintains redaction even when objects are temporarily blocked by other items.
                </p>
              </div>
              <div className="glass-card p-lg rounded-xl hover:bg-surface-variant/30 transition-all group">
                <span className="material-symbols-outlined text-primary text-[32px] mb-md group-hover:scale-110 transition-transform">
                  fingerprint
                </span>
                <h3 className="font-headline-md text-headline-md text-on-surface mb-sm">Smart Re-ID</h3>
                <p className="font-body-md text-body-md text-on-surface-variant">
                  Consistent identity tracking even if a subject leaves and re-enters the frame.
                </p>
              </div>
              <div className="glass-card p-lg rounded-xl hover:bg-surface-variant/30 transition-all group">
                <span className="material-symbols-outlined text-primary text-[32px] mb-md group-hover:scale-110 transition-transform">
                  security
                </span>
                <h3 className="font-headline-md text-headline-md text-on-surface mb-sm">Ghost Box Protection</h3>
                <p className="font-body-md text-body-md text-on-surface-variant">
                  Predictive blurring ensures redaction applied before the subject is fully visible.
                </p>
              </div>
              <div className="glass-card p-lg rounded-xl hover:bg-surface-variant/30 transition-all group">
                <span className="material-symbols-outlined text-primary text-[32px] mb-md group-hover:scale-110 transition-transform">
                  bolt
                </span>
                <h3 className="font-headline-md text-headline-md text-on-surface mb-sm">Real-Time Processing</h3>
                <p className="font-body-md text-body-md text-on-surface-variant">
                  GPU-accelerated inference speeds for near-instant results on large datasets.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Interactive Demo Section */}
        <section className="py-xl px-gutter relative overflow-hidden">
          <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[400px] bg-primary/5 blur-[120px] rounded-full pointer-events-none"></div>
          <div className="max-w-container-max mx-auto text-center">
            <h2 className="font-display text-headline-lg text-on-surface mb-xl">See the AI in Action</h2>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-md items-center bg-surface-container rounded-2xl p-md border border-outline-variant">
              {/* Original */}
              <div className="flex flex-col gap-sm">
                <div className="aspect-video rounded-lg overflow-hidden relative bg-black">
                  <div
                    className="absolute inset-0 bg-cover bg-center"
                    style={{
                      backgroundImage: `url('https://lh3.googleusercontent.com/aida-public/AB6AXuCyNQ0be7krH6JD5kgdJ6gy1UjP15ntl0EDoWqhbF3FI_Xlsov1tG8Lwiz7tQ9vdOZD3SX0ZDk-ii7o8l2ZXvFw33QbFmDbcqdnXWDhXLT4-HE5-1ZZg2wfWKP8T3dBqhqvc_nuVYuwLtn0H8gqs-Q8SVDVw32685AUuO0tS-yR0dyMPEukzLHpbL9xGSHzZwRhzEiIH3n8httxBgGQ856Ga7O8FpPQFZkhmywgmdxSUbOkgNKqHJQHI-YKYkfRTYv0Ptjgqwt_afY')`,
                    }}
                  ></div>
                  <div className="absolute top-sm left-sm bg-black/60 px-sm py-xs rounded font-label-sm text-label-sm text-white">
                    RAW FEED
                  </div>
                </div>
                <p className="font-label-md text-label-md text-on-surface-variant">Original Source</p>
              </div>
              {/* Redacted */}
              <div className="flex flex-col gap-sm">
                <div className="aspect-video rounded-lg overflow-hidden relative bg-black">
                  <div
                    className="absolute inset-0 bg-cover bg-center"
                    style={{
                      backgroundImage: `url('https://lh3.googleusercontent.com/aida-public/AB6AXuCsVlBL5pw650QGPS4b7O9KljNJAHWrpa_iNOJtzuhm5bGsbkuoFJ1CBs3PORlUg0qpdCe8DqOC6RFJh3owa7dz2kZ5uJ7bp2lYz5KQQBg-4pDoPZCq-dwSY6y2oplYU84YdYG-QLiAUWq88I7TMzvz40cAszARx8pPxqLoO4rAziR9xeT_iKzmvz2oMk4xrAHUacXesNnh_ID3H3EmGyGpXtrLykgptRVFZqEUeZUTeZDbV7cFuO8E0IfsErF_Yr8LcWHBYmO9BoQ')`,
                    }}
                  ></div>
                  <div className="absolute top-sm left-sm bg-primary px-sm py-xs rounded font-label-sm text-label-sm text-on-primary">
                    SAFEFRAME PROTECTED
                  </div>
                  <div className="absolute inset-0 border-2 border-primary/20 pointer-events-none"></div>
                </div>
                <p className="font-label-md text-label-md text-primary">AI Redacted Output</p>
              </div>
            </div>
            <div className="mt-xl">
              <button
                type="button"
                onClick={() => navigate('/upload')}
                className="bg-primary text-on-primary px-xl py-md rounded-lg font-label-md text-label-md font-bold hover:bg-primary-container hover:text-on-primary-container transition-all shadow-lg shadow-primary/20 cursor-pointer"
              >
                Try Demo with Your Video
              </button>
              <p className="font-label-sm text-label-sm text-on-surface-variant mt-md">
                Free trial includes up to 60 seconds of HD processing.
              </p>
            </div>
          </div>
        </section>

        {/* Final CTA Section */}
        <section className="py-xl px-gutter">
          <div className="max-w-4xl mx-auto glass-card p-xl rounded-3xl text-center border-primary/20 bg-gradient-to-br from-surface-container to-surface">
            <h2 className="font-display text-display text-on-surface mb-md">Ready to secure your media?</h2>
            <p className="font-body-lg text-body-lg text-on-surface-variant mb-xl max-w-2xl mx-auto">
              Join technical teams at global news agencies, research firms, and legal practices who trust SafeFrame for automated privacy at scale.
            </p>
            <div className="flex flex-col sm:flex-row justify-center gap-md">
              <button
                type="button"
                onClick={() => navigate('/upload')}
                className="bg-on-surface text-surface px-xl py-md rounded-lg font-label-md text-label-md font-bold hover:bg-white transition-all cursor-pointer"
              >
                Create Free Account
              </button>
              <button
                type="button"
                onClick={() => setSalesModalOpen(true)}
                className="border border-outline-variant text-on-surface px-xl py-md rounded-lg font-label-md text-label-md font-bold hover:bg-surface-variant transition-all cursor-pointer"
              >
                Contact Sales
              </button>
            </div>
          </div>
        </section>
      </main>

      {/* Contact Sales Modal */}
      {salesModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-md">
          <div className="glass-card max-w-md w-full p-xl rounded-2xl border border-outline-variant flex flex-col gap-md">
            <div className="flex justify-between items-center">
              <h3 className="font-headline-md text-on-surface font-bold">Contact Enterprise Sales</h3>
              <button
                type="button"
                onClick={() => setSalesModalOpen(false)}
                className="text-on-surface-variant hover:text-on-surface cursor-pointer"
              >
                <span className="material-symbols-outlined">close</span>
              </button>
            </div>
            <p className="font-body-md text-on-surface-variant">
              Get custom high-throughput CUDA deployment, dedicated enterprise SLA, and priority support.
            </p>
            <div className="p-md bg-surface-container rounded-lg border border-outline-variant text-sm font-mono text-primary">
              sales@safeframe.ai
            </div>
            <button
              type="button"
              onClick={() => {
                window.location.href = 'mailto:sales@safeframe.ai?subject=SafeFrame%20Enterprise%20Inquiry';
                setSalesModalOpen(false);
              }}
              className="bg-primary text-on-primary py-md rounded-lg font-label-md font-bold hover:opacity-90 transition-all cursor-pointer"
            >
              Open Email Client
            </button>
          </div>
        </div>
      )}

      {/* Developer API & SDK Modal */}
      {apiModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-md">
          <div className="glass-card max-w-lg w-full p-xl rounded-2xl border border-outline-variant flex flex-col gap-md">
            <div className="flex justify-between items-center">
              <div className="flex items-center gap-sm">
                <span className="material-symbols-outlined text-primary">code</span>
                <h3 className="font-headline-md text-on-surface font-bold">SafeFrame Developer API</h3>
              </div>
              <button
                type="button"
                onClick={() => setApiModalOpen(false)}
                className="text-on-surface-variant hover:text-on-surface cursor-pointer"
              >
                <span className="material-symbols-outlined">close</span>
              </button>
            </div>
            <p className="font-body-md text-on-surface-variant">
              Integrate SafeFrame neural redaction directly into your Python pipelines or REST microservices.
            </p>
            <div className="p-md bg-black/60 rounded-lg border border-outline-variant text-xs font-mono text-primary flex flex-col gap-xs">
              <div><span className="text-on-surface-variant"># Python SDK Quickstart</span></div>
              <div>import safeframe</div>
              <div>client = safeframe.Client(api_key=&quot;sf_live_...&quot;)</div>
              <div>task = client.redact_video(&quot;input.mp4&quot;, targets=[&quot;faces&quot;, &quot;plates&quot;])</div>
              <div>print(task.status) # COMPLETED</div>
            </div>
            <div className="flex justify-end gap-sm pt-sm">
              <button
                type="button"
                onClick={() => setApiModalOpen(false)}
                className="bg-primary text-on-primary px-lg py-sm rounded-lg font-label-md font-bold hover:opacity-90 transition-all cursor-pointer"
              >
                Close Documentation
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Footer */}
      <footer className="bg-surface-container-lowest border-t border-outline-variant">
        <div className="w-full py-xl px-gutter flex flex-col md:flex-row justify-between items-center max-w-container-max mx-auto gap-xl">
          <div className="flex flex-col gap-sm items-center md:items-start">
            <span className="font-display text-headline-md text-on-surface font-bold">SafeFrame</span>
            <p className="font-body-md text-label-sm text-on-surface-variant max-w-xs text-center md:text-left">
              © 2024 SafeFrame AI. Technical Authority in Video Privacy.
            </p>
          </div>
          <div className="flex flex-wrap justify-center gap-lg">
            <button
              type="button"
              onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
              className="text-on-surface-variant font-label-sm text-label-sm hover:text-primary underline transition-colors cursor-pointer"
            >
              Privacy Policy
            </button>
            <button
              type="button"
              onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
              className="text-on-surface-variant font-label-sm text-label-sm hover:text-primary underline transition-colors cursor-pointer"
            >
              Terms of Service
            </button>
            <button
              type="button"
              onClick={() => setSalesModalOpen(true)}
              className="text-on-surface-variant font-label-sm text-label-sm hover:text-primary underline transition-colors cursor-pointer"
            >
              Security
            </button>
            <button
              type="button"
              onClick={() => setSalesModalOpen(true)}
              className="text-on-surface-variant font-label-sm text-label-sm hover:text-primary underline transition-colors cursor-pointer"
            >
              Contact
            </button>
          </div>
          <div className="flex gap-md">
            <button
              type="button"
              onClick={() => setSalesModalOpen(true)}
              className="w-10 h-10 rounded-full border border-outline-variant flex items-center justify-center text-on-surface-variant hover:border-primary hover:text-primary cursor-pointer transition-all"
              title="Global Enterprise Support"
            >
              <span className="material-symbols-outlined text-[18px]">public</span>
            </button>
            <button
              type="button"
              onClick={() => setApiModalOpen(true)}
              className="w-10 h-10 rounded-full border border-outline-variant flex items-center justify-center text-on-surface-variant hover:border-primary hover:text-primary cursor-pointer transition-all"
              title="Developer Console"
            >
              <span className="material-symbols-outlined text-[18px]">terminal</span>
            </button>
          </div>
        </div>
      </footer>
    </div>
  );
};
