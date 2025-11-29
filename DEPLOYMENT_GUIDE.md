# Streamlit Cloud Deployment Guide

## 📱 Access Your App from Any Device (Phone, Tablet, Computer)

Follow these steps to deploy your Candlestick Analysis app to Streamlit Cloud and access it from anywhere!

---

## Step 1: Prepare Your Project

### 1.1 Create a GitHub Account
- Go to [github.com](https://github.com)
- Sign up for a free account if you don't have one

### 1.2 Install Git (if not already installed)
```powershell
# Check if git is installed
git --version

# If not installed, download from: https://git-scm.com/download/win
```

### 1.3 Initialize Git Repository
Open PowerShell in your project folder and run:

```powershell
cd d:/ChartAnalysis/candlestick_project

# Initialize git
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit - Candlestick Analysis App"
```

---

## Step 2: Push to GitHub

### 2.1 Create a New Repository on GitHub
1. Go to [github.com/new](https://github.com/new)
2. Repository name: `candlestick-analysis`
3. Description: `NSE Stock & MCX Commodity Candlestick Pattern Analysis with ML Predictions`
4. Keep it **Public** (required for free Streamlit Cloud)
5. **DO NOT** initialize with README, .gitignore, or license
6. Click **Create repository**

### 2.2 Push Your Code
Copy the commands from GitHub (they'll look like this):

```powershell
git remote add origin https://github.com/YOUR_USERNAME/candlestick-analysis.git
git branch -M main
git push -u origin main
```

**Replace `YOUR_USERNAME` with your actual GitHub username!**

---

## Step 3: Deploy to Streamlit Cloud

### 3.1 Sign Up for Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **Sign up** or **Continue with GitHub**
3. Authorize Streamlit to access your GitHub repositories

### 3.2 Deploy Your App
1. Click **New app** button
2. Fill in the details:
   - **Repository**: Select `YOUR_USERNAME/candlestick-analysis`
   - **Branch**: `main`
   - **Main file path**: `dashboard/app.py`
   - **App URL**: Choose a custom URL (e.g., `candlestick-analysis`)

3. Click **Deploy!**

### 3.3 Wait for Deployment
- First deployment takes 2-5 minutes
- You'll see logs showing the installation progress
- Once complete, your app will be live!

---

## Step 4: Access Your App

### 4.1 Get Your App URL
Your app will be available at:
```
https://YOUR_APP_NAME.streamlit.app
```

For example: `https://candlestick-analysis.streamlit.app`

### 4.2 Access from Any Device
- **Phone**: Open the URL in your mobile browser (Chrome, Safari, etc.)
- **Tablet**: Same URL in tablet browser
- **Computer**: Any web browser
- **Share**: Send the URL to anyone!

### 4.3 Add to Phone Home Screen (Optional)
**For Android:**
1. Open the app URL in Chrome
2. Tap the **⋮** menu
3. Select **Add to Home screen**
4. Name it "Stock Analysis"
5. Tap **Add**

**For iPhone:**
1. Open the app URL in Safari
2. Tap the **Share** button
3. Select **Add to Home Screen**
4. Name it "Stock Analysis"
5. Tap **Add**

Now it works like a native app! 📱

---

## Step 5: Important Notes

### ⚠️ Limitations of Free Streamlit Cloud

1. **Data Storage**: 
   - Your data files are NOT included in the deployment (they're in `.gitignore`)
   - The app will download fresh data when first accessed
   - Data is temporary and resets when the app restarts

2. **Resources**:
   - 1 GB RAM limit
   - Apps sleep after inactivity (wake up when accessed)
   - Limited to 3 apps on free tier

3. **Model Files**:
   - Pre-trained models are NOT deployed (too large)
   - You'll need to train a model after deployment OR
   - Use a smaller model that can be committed to git

### 💡 Solutions

**Option A: Auto-download data on startup**
The app already has logic to download data when missing. First load will be slow but subsequent loads will be faster.

**Option B: Use sample data**
Create a small sample dataset that can be committed to git for demo purposes.

**Option C: Train model on cloud**
Add a button in the app to train the model using cloud resources (takes time but works).

---

## Step 6: Update Your App

Whenever you make changes locally:

```powershell
cd d:/ChartAnalysis/candlestick_project

# Add changes
git add .

# Commit with a message
git commit -m "Updated prediction logic"

# Push to GitHub
git push
```

Streamlit Cloud will **automatically redeploy** your app within 1-2 minutes!

---

## Troubleshooting

### App Won't Start
- Check the logs in Streamlit Cloud dashboard
- Ensure `requirements.txt` has all dependencies
- Verify `dashboard/app.py` path is correct

### Data Not Loading
- The app will try to download data on first run
- This is normal and expected
- Wait 2-3 minutes for initial data download

### Model Not Found
- Train the model using the app interface OR
- Remove model-dependent features temporarily

### App is Slow
- Free tier has limited resources
- Consider reducing the number of stocks downloaded
- Optimize feature engineering

---

## 🎉 Success!

You now have a **cloud-hosted stock analysis app** accessible from:
- ✅ Your Android phone
- ✅ Your iPhone
- ✅ Your tablet
- ✅ Any computer
- ✅ Shareable with anyone!

**No APK needed - just a URL!** 🚀

---

## Next Steps

1. **Customize**: Add more features, improve UI
2. **Share**: Send the URL to friends/colleagues
3. **Monitor**: Check Streamlit Cloud dashboard for usage stats
4. **Upgrade**: Consider Streamlit Cloud Pro for more resources ($20/month)

---

## Support

- **Streamlit Docs**: [docs.streamlit.io](https://docs.streamlit.io)
- **Community**: [discuss.streamlit.io](https://discuss.streamlit.io)
- **GitHub Issues**: Create issues in your repository

---

**Happy Trading! 📈📉**
