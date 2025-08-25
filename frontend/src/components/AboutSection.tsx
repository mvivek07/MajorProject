
import React from 'react';

const AboutSection: React.FC = () => {
  const technologies = [
    "scikit-learn", "pandas", "TensorFlow", "PyTorch", "LSTM Networks", 
    "Random Forest", "XGBoost", "AWS", "GCP", "Azure ML"
  ];

  return (
    <section id="about" className="section-padding">
      <div className="container mx-auto px-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-12 items-center">
          <div className="animate-fade-in">
            <h2 className="text-3xl md:text-4xl font-bold mb-6">About CashWise AI</h2>
            <p className="text-lg text-gray-600 mb-6">
              We're a team of data scientists, financial analysts, and software engineers passionate about making sophisticated financial forecasting accessible to everyone.
            </p>
            <p className="text-lg text-gray-600 mb-6">
              Our mission is to eliminate financial uncertainty for businesses and individuals through the power of machine learning and artificial intelligence.
            </p>
            <p className="text-lg text-gray-600">
              Founded in 2023, CashWise AI has already helped thousands of users gain clarity on their financial future and make better financial decisions.
            </p>
          </div>
          
          <div className="animate-fade-in [animation-delay:200ms]">
            <div className="bg-blue-50 rounded-lg p-8">
              <h3 className="text-2xl font-bold mb-6">Our ML Tech Stack</h3>
              
              <div className="flex flex-wrap gap-2">
                {technologies.map((tech, index) => (
                  <div 
                    key={index} 
                    className="bg-white px-4 py-2 rounded-full text-sm font-medium border shadow-sm"
                  >
                    {tech}
                  </div>
                ))}
              </div>
              
              <div className="mt-8 p-4 bg-cashwise-blue-light/30 rounded-lg">
                <p className="text-sm">
                  Our machine learning models are constantly improving, with a 92% accuracy rate in predicting cash flow patterns up to 90 days in advance.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default AboutSection;
