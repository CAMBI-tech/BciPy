clear all
%close all
clear classes

mod1 = py.importlib.import_module('function_density_estimation');
py.reload(mod1);
mod = py.importlib.import_module('demo');
py.reload(mod);

num_samples = 300;
mu_1 = -2;
sig_1 = 2;
mu_2 = 4;
sig_2 = 1;

x1=  mu_1 + sig_1 * randn(num_samples,1);
x2=  mu_2 + sig_2 * randn(num_samples,1);
x = [x1;x2];
hop = 0.01;

x_lsp = [-10:hop:10].';
pdf_1 = normpdf(x_lsp,mu_1,sig_1);
pdf_2 = normpdf(x_lsp,mu_2,sig_2);
true_pdf = (pdf_1 + pdf_2)/(sum(pdf_1+pdf_2)*hop);

kde_matlab = kde1d(x);
p_matlab = kde_matlab.probs(x_lsp);

p_py = py.demo.test_kde(toggleNumpy(x),toggleNumpy(x_lsp));
p_py = exp(toggleNumpy(p_py));

figure()
subplot(2,1,1)
plot(x_lsp,true_pdf,'k','linewidth',2)
hold()
plot(x_lsp,p_matlab,'b','linewidth',2)
plot(x_lsp,p_py,'--r','linewidth',2)

plot(x, zeros(size(x)),'k*');
legend('true\_pdf','matlab\_kde','py\_kde','samples')
xlabel('x')
ylabel('p(x)')
title('Estimate using: Silverman density estimation')

