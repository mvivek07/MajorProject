
import React from 'react';
import { Store, UserCircle, Building2 } from "lucide-react";

const UseCasesSection: React.FC = () => {
  const useCases = [
    {
      icon: <Store className="h-8 w-8 text-cashwise-blue" />,
      title: "Small & Medium Businesses",
      description: "Avoid cash shortages, make informed growth decisions, and ensure your business stays financially healthy.",
      benefits: ["Predict cash shortages 30-90 days in advance", "Plan equipment purchases confidently", "Optimize invoice timing"]
    },
    {
      icon: <UserCircle className="h-8 w-8 text-cashwise-blue" />,
      title: "Freelancers & Individuals",
      description: "Get clarity on your finances, budget effectively, and ensure stability through income fluctuations.",
      benefits: ["Budget monthly expenses accurately", "Plan for taxes and large purchases", "Visual forecast of future cash position"]
    },
    {
      icon: <Building2 className="h-8 w-8 text-cashwise-blue" />,
      title: "CFOs & Financial Teams",
      description: "Make data-driven decisions with advanced analytics and real-time visibility into your company's financial future.",
      benefits: ["Scenario planning tools", "Board-ready financial projections", "Integration with existing systems"]
    }
  ];

  return (
    <section id="features" className="section-padding bg-gradient-to-b from-white to-blue-50">
      <div className="container mx-auto px-4">
        <div className="text-center max-w-3xl mx-auto mb-16">
          <h2 className="text-3xl md:text-4xl font-bold">
            Tailored Cash Flow Management for Every Need
          </h2>
          <p className="mt-4 text-lg text-gray-600">
            Our AI-driven platform adapts to various use cases, providing personalized insights for different financial scenarios.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {useCases.map((useCase, index) => (
            <div 
              key={index} 
              className="feature-card flex flex-col animate-fade-in"
              style={{ animationDelay: `${index * 150}ms` }}
            >
              <div className="mb-4">
                <div className="h-14 w-14 rounded-full bg-cashwise-blue-light/30 flex items-center justify-center">
                  {useCase.icon}
                </div>
              </div>
              <h3 className="text-xl font-bold mb-3">{useCase.title}</h3>
              <p className="text-gray-600 mb-4">{useCase.description}</p>
              <ul className="mt-auto space-y-2">
                {useCase.benefits.map((benefit, i) => (
                  <li key={i} className="flex items-start">
                    <div className="h-5 w-5 rounded-full bg-cashwise-green-light flex items-center justify-center mr-2 mt-0.5">
                      <svg width="12" height="12" viewBox="0 0 12 12" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M10 3L4.5 8.5L2 6" stroke="#69C77F" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                      </svg>
                    </div>
                    <span className="text-sm">{benefit}</span>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default UseCasesSection;
