High pass filter

https://www.researchgate.net/publication/273405257_How_inappropriate_high-pass_filters_can_produce_artifactual_effects_and_incorrect_conclusions_in_ERP_studies_of_language_and_cognition (old luck)

https://pdf.sciencedirectassets.com/271055/1-s2.0-S0165027021X00027/1-s2.0-S0165027021000157/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjECwaCXVzLWVhc3QtMSJIMEYCIQDxTd8JwZN48ThLFmYagqTpA6p8Tx5H7Lru3%2BjRYlnr5AIhALNrs6A3p0upq6SBfvqRrwK%2FwXwyZU2PWSjG19zB9quoKrsFCIX%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQBRoMMDU5MDAzNTQ2ODY1IgzDCEwy2efaHKnxScsqjwVjB%2BUsBZccowt96immEwPJUDYIvWkAZpB%2FyS6VdR1PnFs%2FCFJf7OEbJqnSExuaBnGRFuH9WXGND%2FVY0IFfw4UaGQdbV6EvdlD1kweh4aLKgk%2FG0K3eGhXAyiyte64oiY1fu6EBVA5PxuS%2Fr3Pzd9qwX1uJ6zE2b0XR%2FE5gBVeCoVYuwDwSP3g745IHxiBEMVHzQ407a%2FVjQg%2FKCxgSArHHmNiNYkfAmVdJmxOxhmTL7JGVzCc06UEgskCjvppcEdj06cYdvD%2F6%2Ff8CA7DQjt1qlomxuLa7Oct3KZYqtpUrpKWLHAYASUGipIv6HopYihqDzZbBMnCbgHQvesim8FB3%2B%2BQewPuDqbAecGzukgb%2FnOZT70fMS2HsJAgbW%2FGPVW0EIUYxTkuAoG6MCpl9R8a0MgS0Ds%2Fx2oVodlVet2qXOp3Kxfy2PttM%2B4ND5NxL0ZEKC42tRx%2FtcGlYJrtoaNeIcGq1ToRFC%2BQ19M2z4GpR9FdPXxaVlRoSJCklwi4HFiR8IhRB44JWTFfzWQmf0IACaxoHoPkyzUw%2BDbQr8hL5GElg80166gyESM6N5nIa9jNTohXgn7etJnZWouxxdFWQ8RTeD2IJBQt64HxvptXoRq%2BXfQUi07UimdOcQJ3WFPBsuSA7UbrPbb%2B7LXbwtfLuqNh6bA7pfT9h39MJ9fLva1mGK4Wv6D3MukHGq3Mxpv7mQWuOAPRO%2BsYEjW0k8sY9pUCp%2Bzxa0qIB3Zotx0paKtcmMxWwHcUcCfCeclDvC6c%2FgCgsQHomYK9g0NwN2HzjEz3skOlChASIa9OKL8G1VPYaTj5sR8IgSUgFRl%2FHQkTEZ76CcBnfnYUjdAHSg1aMbkPxd7M8W5cb68b7tLwUMNnziqsGOrABDeOIE75kUdfhyc4dBdmkU30jaRt%2FCRwdPyyx8ItTDMKnfDJnyqm4Sx%2Be6u%2FzbZofLDDoDHyPg8txSPD6HetbsGDa68G%2FY1%2FNy7xhI6xGQ1pIy7HX0k%2BKceKMGjXqCgQVwBEkLRoxNyoL7JB2lKI%2F2ZqnG16iUlobUskCR8k2vceRaK1zI3Dy%2Bcr5EudJ2dT81WhVBNns3002NBYjbrMffuk2K0nP%2FDXFkXAdbLQthB8%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20231126T040218Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYROBTVQKD%2F20231126%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=2d361e1b7ed5b0c90784b118046221e4d5a31f8541520ad9c4c1115356fee4b8&hash=006fe519220eea4a7ca609d37ca5a10e94752b2bef2b05da3d0f8427aa96edda&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0165027021000157&tid=spdf-bb36de1b-f0fc-499d-9714-10c0527bfe74&sid=011c94e671422242b67875f4bff9412f8998gxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=0f165c510d0651045d&rr=82bf3c31bfe75eec&cc=us (different paper)
 
https://www.nature.com/articles/s41598-023-27528-0 ****

When resampling 50 data trials instead of using all the trials to compute significance, we observed that rejecting trials led to significant improvement for most methods, meaning that these methods are indeed capable of removing bad trials. For example, using a 50 data trials bootstrap with ASR, a threshold of 10 increased performance for the Face and Oddball datasets (p < 0.0001 in both cases), although about 60% to 80% of trials were rejected. Using a 50 data trials bootstrap, Brainstorm sensitivity 5 for high-frequency artifacts provided significant improvements (p < 0.004 in all datasets with 19% to 45% of trials rejected), and MNE Autoreject also provided significant improvements (Go/No-go: p = 0.005 and 12% of trials rejected; Oddball: p < 0.001 and 19% of trials rejected). However, as seen in Table 1, this was not the case when bootstrapping all remating trials: the removal of bad trials most often failed to compensate for the decrease in the number of trials and associated decrease in statistical power compared to the control condition where no trials were removed.
"However, as seen in Table 1, this was not the case when bootstrapping all remating trials: the removal of bad trials most often failed to compensate for the decrease in the number of trials and associated decrease in statistical power compared to the control condition where no trials were removed."