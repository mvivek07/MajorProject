import React, { useState, useEffect } from 'react';
import { useAuth } from '@/contexts/AuthContext';
import { useNavigate } from 'react-router-dom';

// Define a type for the API response for better type safety.
interface AuthResponse {
  access_token?: string;
  token_type?: string;
  detail?: string;
}

const AuthPage: React.FC = () => {
  const { login } = useAuth();
  const navigate = useNavigate();

  // State to toggle between Sign In and Register forms
  const [isRegistering, setIsRegistering] = useState<boolean>(false);

  // --- Form Input States ---
  const [email, setEmail] = useState<string>('');
  const [password, setPassword] = useState<string>('');
  const [confirmPassword, setConfirmPassword] = useState<string>('');
  const [fullName, setFullName] = useState<string>('');
  const [companyName, setCompanyName] = useState<string>('');
  const [jobTitle, setJobTitle] = useState<string>('');
  const [industry, setIndustry] = useState<string>('');
  const [annualRevenue, setAnnualRevenue] = useState<string>('');
  const [primaryFinancialGoal, setPrimaryFinancialGoal] = useState<string>('');

  // --- UI/UX States ---
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  // This useEffect hook handles the redirection after a successful login.
  useEffect(() => {
    if (successMessage && successMessage.includes('Redirecting')) {
      const timer = setTimeout(() => {
        navigate('/');
      }, 1000); // Wait 1 second to show the message before redirecting

      // Cleanup the timer if the component unmounts
      return () => clearTimeout(timer);
    }
  }, [successMessage, navigate]);

  /**
   * Handles the form submission for both registration and login.
   */
  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);
    setSuccessMessage(null);

    // Client-side validation for registration
    if (isRegistering) {
      if (password !== confirmPassword) {
        setError('Passwords do not match.');
        setIsLoading(false);
        return;
      }
      if (password.length < 8) {
        setError('Password must be at least 8 characters long.');
        setIsLoading(false);
        return;
      }
    }

    const endpoint = isRegistering ? '/api/users/register' : '/api/token';
    let body;

    if (isRegistering) {
      body = JSON.stringify({
        email,
        password,
        full_name: fullName,
        company_name: companyName,
        job_title: jobTitle,
        industry,
        annual_revenue: annualRevenue,
        primary_financial_goal: primaryFinancialGoal,
      });
    } else {
      const formData = new URLSearchParams();
      formData.append('username', email);
      formData.append('password', password);
      body = formData;
    }

    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': isRegistering ? 'application/json' : 'application/x-www-form-urlencoded',
        },
        body: body,
      });

      const data: AuthResponse = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'An unknown error occurred.');
      }

      if (isRegistering) {
        setSuccessMessage('Registration successful! Please sign in.');
        setIsRegistering(false);
        clearAllFields();
      } else {
        // On successful login, save the token and set the success message
        if (data.access_token) {
          login(data.access_token);
          setSuccessMessage('Sign in successful! Redirecting...');
        }
      }

    } catch (err: any) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };
  
  const clearAllFields = () => {
      setEmail('');
      setPassword('');
      setConfirmPassword('');
      setFullName('');
      setCompanyName('');
      setJobTitle('');
      setIndustry('');
      setAnnualRevenue('');
      setPrimaryFinancialGoal('');
      setError(null);
      setSuccessMessage(null);
  }

  const toggleForm = () => {
    setIsRegistering(!isRegistering);
    clearAllFields();
  };

  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-100 font-sans py-12">
      <div className="w-full max-w-md p-8 space-y-6 bg-white rounded-xl shadow-lg">
        <div>
          <h1 className="text-3xl font-bold text-center text-gray-800">
            {isRegistering ? 'Unlock Financial Clarity' : 'Welcome Back'}
          </h1>
          <p className="text-center text-gray-500">
            {isRegistering
              ? 'Join our AI-driven debt optimization platform.'
              : 'Sign in to manage your financial recovery.'}
          </p>
        </div>

        {error && <div className="p-3 text-sm text-center text-red-800 bg-red-100 rounded-lg" role="alert">{error}</div>}
        {successMessage && <div className="p-3 text-sm text-center text-green-800 bg-green-100 rounded-lg" role="alert">{successMessage}</div>}

        <form className="space-y-4" onSubmit={handleSubmit}>
          {isRegistering && (
            <>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label htmlFor="fullName" className="text-sm font-medium text-gray-700">Your Full Name</label>
                  <input id="fullName" type="text" required value={fullName} onChange={(e) => setFullName(e.target.value)} className="w-full px-4 py-2 mt-1 input-style" placeholder="John Doe" />
                </div>
                <div>
                  <label htmlFor="jobTitle" className="text-sm font-medium text-gray-700">Your Role</label>
                  <input id="jobTitle" type="text" required value={jobTitle} onChange={(e) => setJobTitle(e.target.value)} className="w-full px-4 py-2 mt-1 input-style" placeholder="Finance Manager" />
                </div>
              </div>
               <div>
                  <label htmlFor="companyName" className="text-sm font-medium text-gray-700">Company Name</label>
                  <input id="companyName" type="text" required value={companyName} onChange={(e) => setCompanyName(e.target.value)} className="w-full px-4 py-2 mt-1 input-style" placeholder="Your Company Inc." />
                </div>
              <div>
                <label htmlFor="industry" className="text-sm font-medium text-gray-700">Industry</label>
                <select id="industry" required value={industry} onChange={(e) => setIndustry(e.target.value)} className="w-full px-4 py-2 mt-1 input-style">
                  <option value="" disabled>Select your industry</option>
                  <option value="tech">Technology</option>
                  <option value="retail">Retail</option>
                  <option value="manufacturing">Manufacturing</option>
                  <option value="healthcare">Healthcare</option>
                  <option value="services">Professional Services</option>
                  <option value="other">Other</option>
                </select>
              </div>
              <div>
                <label htmlFor="annualRevenue" className="text-sm font-medium text-gray-700">Approx. Annual Revenue</label>
                <select id="annualRevenue" required value={annualRevenue} onChange={(e) => setAnnualRevenue(e.target.value)} className="w-full px-4 py-2 mt-1 input-style">
                  <option value="" disabled>Select a range</option>
                  <option value="0-1M">$0 - $1M</option>
                  <option value="1M-5M">$1M - $5M</option>
                  <option value="5M-20M">$5M - $20M</option>
                  <option value="20M+">$20M+</option>
                </select>
              </div>
              <div>
                <label htmlFor="primaryFinancialGoal" className="text-sm font-medium text-gray-700">Primary Financial Goal</label>
                <select id="primaryFinancialGoal" required value={primaryFinancialGoal} onChange={(e) => setPrimaryFinancialGoal(e.target.value)} className="w-full px-4 py-2 mt-1 input-style">
                  <option value="" disabled>Select your main goal</option>
                  <option value="reduce_debt">Reduce Debt</option>
                  <option value="improve_cash_flow">Improve Cash Flow</option>
                  <option value="optimize_spending">Optimize Spending</option>
                  <option value="secure_funding">Secure Funding</option>
                </select>
              </div>
            </>
          )}

          <div>
            <label htmlFor="email" className="text-sm font-medium text-gray-700">Email Address</label>
            <input id="email" type="email" autoComplete="email" required value={email} onChange={(e) => setEmail(e.target.value)} className="w-full px-4 py-2 mt-1 input-style" placeholder="you@example.com" />
          </div>
          <div>
            <label htmlFor="password"className="text-sm font-medium text-gray-700">Password</label>
            <input id="password" type="password" required value={password} onChange={(e) => setPassword(e.target.value)} className="w-full px-4 py-2 mt-1 input-style" placeholder="••••••••" />
             {isRegistering && <p className="text-xs text-gray-500 mt-1">Must be at least 8 characters.</p>}
          </div>
           {isRegistering && (
            <div>
              <label htmlFor="confirmPassword"className="text-sm font-medium text-gray-700">Confirm Password</label>
              <input id="confirmPassword" type="password" required value={confirmPassword} onChange={(e) => setConfirmPassword(e.target.value)} className="w-full px-4 py-2 mt-1 input-style" placeholder="••••••••" />
            </div>
           )}

          <div>
            <button type="submit" disabled={isLoading} className="w-full flex justify-center py-3 px-4 border border-transparent rounded-lg shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:bg-indigo-400 disabled:cursor-not-allowed">
              {isLoading ? 'Processing...' : (isRegistering ? 'Create Account' : 'Sign In')}
            </button>
          </div>
        </form>

        <div className="text-sm text-center">
          <button onClick={toggleForm} className="font-medium text-indigo-600 hover:text-indigo-500">
            {isRegistering ? 'Already have an account? Sign In' : "Don't have an account? Register"}
          </button>
        </div>
      </div>
       <style>{`
        .input-style {
          background-color: #F3F4F6;
          border: 1px solid transparent;
          border-radius: 0.5rem;
          color: #374151;
          font-size: 1rem;
        }
        .input-style:focus {
          outline: none;
          border-color: #6366F1;
          box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.5);
        }
      `}</style>
    </div>
  );
};

export default AuthPage;
