
import React, { useState } from 'react';
import { ChevronLeft, ChevronRight, Quote } from "lucide-react";
import { Button } from "@/components/ui/button";

const TestimonialsSection: React.FC = () => {
  const testimonials = [
    {
      quote: "CashWise AI has completely transformed how we manage our finances. We can now predict cash crunches weeks in advance and take proactive steps.",
      name: "Sarah Johnson",
      role: "COO at TechStart Inc.",
      image: "/placeholder.svg"
    },
    {
      quote: "As a freelancer with irregular income, this tool has been invaluable. I can finally plan ahead with confidence and know exactly where I stand financially.",
      name: "Michael Chen",
      role: "Independent Designer",
      image: "/placeholder.svg"
    },
    {
      quote: "The machine learning predictions are remarkably accurate. Our board is impressed with the financial visibility we now have across our 12 locations.",
      name: "Rebecca Torres",
      role: "CFO at RetailPlus",
      image: "/placeholder.svg"
    },
  ];

  const [current, setCurrent] = useState(0);

  const next = () => {
    setCurrent((current + 1) % testimonials.length);
  };

  const prev = () => {
    setCurrent((current - 1 + testimonials.length) % testimonials.length);
  };

  return (
    <section className="section-padding bg-white">
      <div className="container mx-auto px-4">
        <div className="text-center max-w-3xl mx-auto mb-12">
          <h2 className="text-3xl md:text-4xl font-bold">What Our Users Say</h2>
        </div>
        
        <div className="relative max-w-4xl mx-auto">
          <div className="testimonial-card p-8 md:p-12">
            <div className="flex justify-center mb-8">
              <div className="h-16 w-16 rounded-full bg-cashwise-blue-light/30 flex items-center justify-center">
                <Quote className="h-8 w-8 text-cashwise-blue" />
              </div>
            </div>
            
            <div className="text-center">
              <blockquote className="text-xl md:text-2xl italic mb-6">
                "{testimonials[current].quote}"
              </blockquote>
              
              <div className="flex items-center justify-center mb-2">
                <div className="w-12 h-12 rounded-full overflow-hidden mr-4">
                  <img 
                    src={testimonials[current].image} 
                    alt={testimonials[current].name}
                    className="w-full h-full object-cover"
                  />
                </div>
                <div className="text-left">
                  <div className="font-bold">{testimonials[current].name}</div>
                  <div className="text-sm text-gray-600">{testimonials[current].role}</div>
                </div>
              </div>
            </div>
          </div>
          
          <div className="flex justify-center mt-8">
            <div className="flex space-x-2">
              {testimonials.map((_, index) => (
                <button
                  key={index}
                  onClick={() => setCurrent(index)}
                  className={`w-3 h-3 rounded-full ${
                    current === index ? 'bg-cashwise-blue' : 'bg-gray-300'
                  }`}
                  aria-label={`Go to testimonial ${index + 1}`}
                />
              ))}
            </div>
          </div>
          
          <div className="absolute top-1/2 -left-4 md:-left-8 transform -translate-y-1/2">
            <Button
              variant="outline" 
              size="icon"
              onClick={prev}
              className="h-10 w-10 rounded-full bg-white shadow-md"
            >
              <ChevronLeft className="h-5 w-5" />
              <span className="sr-only">Previous testimonial</span>
            </Button>
          </div>
          
          <div className="absolute top-1/2 -right-4 md:-right-8 transform -translate-y-1/2">
            <Button
              variant="outline" 
              size="icon"
              onClick={next}
              className="h-10 w-10 rounded-full bg-white shadow-md"
            >
              <ChevronRight className="h-5 w-5" />
              <span className="sr-only">Next testimonial</span>
            </Button>
          </div>
        </div>
      </div>
    </section>
  );
};

export default TestimonialsSection;
