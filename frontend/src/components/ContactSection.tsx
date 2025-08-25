
import React, { useState } from 'react';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { useToast } from "@/components/ui/use-toast";

const ContactSection: React.FC = () => {
  const { toast } = useToast();
  const [email, setEmail] = useState('');
  const [message, setMessage] = useState('');
  const [name, setName] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  
  const handleNewsletterSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    
    // Simulate API call
    setTimeout(() => {
      toast({
        title: "Success!",
        description: "You've been added to our newsletter.",
      });
      setEmail('');
      setIsSubmitting(false);
    }, 1000);
  };
  
  const handleContactSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    
    // Simulate API call
    setTimeout(() => {
      toast({
        title: "Message sent!",
        description: "We'll get back to you as soon as possible.",
      });
      setName('');
      setEmail('');
      setMessage('');
      setIsSubmitting(false);
    }, 1000);
  };

  return (
    <section className="section-padding bg-gradient-to-b from-white to-blue-50">
      <div className="container mx-auto px-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-12">
          {/* Newsletter Signup */}
          <div className="animate-fade-in">
            <div className="bg-white p-8 rounded-lg border shadow-sm">
              <h3 className="text-2xl font-bold mb-4">Stay Updated</h3>
              <p className="text-gray-600 mb-6">
                Subscribe to our newsletter for cash flow management tips, product updates, and industry insights.
              </p>
              
              <form onSubmit={handleNewsletterSubmit}>
                <div className="flex flex-col sm:flex-row gap-3">
                  <Input
                    type="email"
                    placeholder="Enter your email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    required
                    className="flex-grow"
                  />
                  <Button type="submit" disabled={isSubmitting}>
                    {isSubmitting ? "Subscribing..." : "Subscribe"}
                  </Button>
                </div>
              </form>
              
              <p className="mt-4 text-xs text-gray-500">
                We respect your privacy. Unsubscribe at any time.
              </p>
            </div>
          </div>
          
          {/* Contact Form */}
          <div className="animate-fade-in [animation-delay:200ms]">
            <div className="bg-white p-8 rounded-lg border shadow-sm">
              <h3 className="text-2xl font-bold mb-4">Get in Touch</h3>
              <p className="text-gray-600 mb-6">
                Have questions about CashWise AI? Fill out the form below and we'll get back to you.
              </p>
              
              <form onSubmit={handleContactSubmit} className="space-y-4">
                <div>
                  <Input
                    placeholder="Your name"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    required
                  />
                </div>
                
                <div>
                  <Input
                    type="email"
                    placeholder="Your email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    required
                  />
                </div>
                
                <div>
                  <Textarea
                    placeholder="Your message"
                    value={message}
                    onChange={(e) => setMessage(e.target.value)}
                    required
                    rows={4}
                  />
                </div>
                
                <Button type="submit" className="w-full" disabled={isSubmitting}>
                  {isSubmitting ? "Sending..." : "Send Message"}
                </Button>
              </form>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default ContactSection;
