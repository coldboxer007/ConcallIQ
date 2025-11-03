# 🚀 ConcallIQ Deployment Guide

## Pre-Deployment Checklist

### ✅ Files Ready for Deployment
- [x] `app.py` - Main application with secure API key handling
- [x] `requirements.txt` - All Python dependencies listed
- [x] `.gitignore` - Prevents committing sensitive files
- [x] `.streamlit/secrets.toml.example` - Template for secrets configuration
- [x] `README.md` - Complete documentation with deployment instructions

### ⚠️ Before Pushing to GitHub

1. **Verify .gitignore includes:**
   - `.env`
   - `.streamlit/secrets.toml`
   - `__pycache__/`

2. **Remove any hardcoded API keys from:**
   - `app.py` ✅ (Using st.secrets)
   - Configuration files ✅

3. **Test locally one more time:**
   ```bash
   streamlit run app.py
   ```

## Streamlit Cloud Deployment Steps

### Step 1: Push to GitHub
```bash
# Check what will be committed
git status

# Add files (ensure .env is NOT listed)
git add .

# Commit changes
git commit -m "Deploy ConcallIQ v1.0 - Production ready"

# Push to GitHub
git push origin main
```

### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click **"New app"**
4. Configure deployment:
   - **Repository:** Your GitHub repo
   - **Branch:** `main`
   - **Main file path:** `app.py`

### Step 3: Configure Secrets

In Streamlit Cloud app settings → **Secrets**, add:

```toml
GOOGLE_API_KEY = "your-actual-google-gemini-api-key"
```

### Step 4: Deploy!

Click **"Deploy"** and wait for the build to complete.

Your app will be available at:
```
https://[your-username]-concalliq.streamlit.app
```

## Post-Deployment Verification

### Test These Features:
- [ ] Upload a PDF document
- [ ] Generate executive summary
- [ ] View retrieval index analytics
- [ ] Ask questions using Q&A interface
- [ ] Check all expanders work (Context, Debug panels)
- [ ] Verify styling and colors are correct
- [ ] Test sidebar controls
- [ ] Download summary button

## Troubleshooting

### Issue: "API Key not found" error
**Solution:** Check Streamlit Cloud secrets are properly configured.

### Issue: Dependencies fail to install
**Solution:** Verify `requirements.txt` has correct package versions.

### Issue: PDF upload fails
**Solution:** Check file size limits on Streamlit Cloud (typically 200MB max).

### Issue: Memory errors during processing
**Solution:** Streamlit Cloud free tier has 1GB memory. Optimize chunk sizes if needed.

## Environment Variables

The app supports two methods for API key configuration:

### Local Development
Uses `.env` file:
```bash
GOOGLE_API_KEY="your-key-here"
```

### Production (Streamlit Cloud)
Uses Streamlit Secrets (configured in dashboard):
```toml
GOOGLE_API_KEY = "your-key-here"
```

The app automatically detects the environment and uses the appropriate method.

## Security Best Practices

✅ **DO:**
- Use Streamlit secrets for production
- Keep `.env` in `.gitignore`
- Rotate API keys regularly
- Monitor API usage in Google Cloud Console

❌ **DON'T:**
- Commit `.env` files
- Hardcode API keys in code
- Share secrets in public repositories
- Use the same API key for dev and prod

## Support

For deployment issues:
- Check Streamlit Community: https://discuss.streamlit.io
- Review logs in Streamlit Cloud dashboard
- Verify Google Gemini API quotas

---

**Ready to deploy?** Follow the steps above and your ConcallIQ app will be live in minutes! 🎉
